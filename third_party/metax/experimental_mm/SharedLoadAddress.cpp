#include "MatmulPipeline.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

static cl::opt<std::string> Input(cl::Positional, cl::Required);
static cl::opt<std::string> Output("o", cl::Required);
static cl::opt<bool> BaseZero("dynamic-shared-base-zero", cl::init(false));
static cl::opt<unsigned> GroupRows("group-rows", cl::init(1));
static cl::opt<unsigned> Splits("split-partial-i32", cl::init(1));
static cl::list<unsigned> LayoutShape("layout-shape", cl::CommaSeparated);
static cl::opt<bool> MatmulPipeline("matmul-pipeline", cl::init(false));

struct OffsetExpression {
  APInt constant = APInt(32, 0);
  SmallMapVector<Value *, APInt, 4> terms;
};

// The caller must establish the target ABI contract. A zero-sized declaration
// alone does not prove the runtime address of dynamic shared memory.
static GlobalVariable *findDynamicShared(Module &module) {
  GlobalVariable *root = nullptr;
  for (GlobalVariable &global : module.globals()) {
    if (global.getAddressSpace() != 3)
      continue;
    if (root)
      return nullptr;
    auto *array = dyn_cast<ArrayType>(global.getValueType());
    if (!array || array->getNumElements() != 0 ||
        !array->getElementType()->isIntegerTy(8) || !global.isDeclaration() ||
        !global.hasExternalLinkage())
      return nullptr;
    root = &global;
  }
  return root;
}

static bool collectOffset(Value *pointer, GlobalVariable *root,
                          const DataLayout &layout, OffsetExpression &offset,
                          unsigned depth = 0) {
  if (pointer == root)
    return true;
  if (depth == 32)
    return false;
  auto *gep = dyn_cast<GEPOperator>(pointer);
  if (!gep || gep->getPointerAddressSpace() != 3 ||
      !collectOffset(gep->getPointerOperand(), root, layout, offset, depth + 1))
    return false;
  SmallMapVector<Value *, APInt, 4> terms;
  APInt constant(32, 0);
  if (!gep->collectOffset(layout, 32, terms, constant))
    return false;
  offset.constant += constant;
  for (auto &term : terms) {
    auto inserted = offset.terms.try_emplace(term.first, APInt(32, 0));
    inserted.first->second += term.second;
  }
  return true;
}

static unsigned rewriteSharedLoads(Module &module) {
  const DataLayout &layout = module.getDataLayout();
  GlobalVariable *root = findDynamicShared(module);
  if (!BaseZero || !root || layout.getPointerSizeInBits(3) != 32 ||
      layout.getIndexSizeInBits(3) != 32 || layout.isNonIntegralAddressSpace(3))
    return 0;

  SmallVector<Use *> uses;
  for (Function &function : module) {
    for (Instruction &instruction : instructions(function)) {
      if (auto *load = dyn_cast<LoadInst>(&instruction)) {
        if (!load->isVolatile() && !load->isAtomic() &&
            load->getPointerAddressSpace() == 3)
          uses.push_back(&load->getOperandUse(0));
      } else if (auto *call = dyn_cast<CallInst>(&instruction)) {
        Function *callee = call->getCalledFunction();
        if (callee && callee->getName().starts_with("llvm.mxc.lds.") &&
            call->arg_size() == 1 &&
            call->getArgOperand(0)->getType()->isPointerTy() &&
            call->getArgOperand(0)->getType()->getPointerAddressSpace() == 3)
          uses.push_back(&call->getArgOperandUse(0));
      }
    }
  }

  DenseMap<Value *, Value *> replacements;
  SmallVector<Instruction *> oldPointers;
  unsigned changed = 0;
  for (Use *use : uses) {
    Value *pointer = use->get();
    auto found = replacements.find(pointer);
    if (found != replacements.end()) {
      use->set(found->second);
      ++changed;
      continue;
    }
    // Placing the replacement at the original definition retains dominance
    // and loop-invariant address placement without scheduling memory accesses.
    auto *definition = dyn_cast<GetElementPtrInst>(pointer);
    if (!definition)
      continue;
    OffsetExpression offset;
    if (!collectOffset(pointer, root, layout, offset))
      continue;
    IRBuilder<> builder(definition);
    Value *bytes = builder.getInt32(offset.constant.getZExtValue());
    for (auto &term : offset.terms) {
      if (term.second.isZero())
        continue;
      Value *index =
          builder.CreateSExtOrTrunc(term.first, builder.getInt32Ty());
      if (!term.second.isOne())
        index = builder.CreateMul(
            index, ConstantInt::get(builder.getInt32Ty(), term.second));
      // GEP index arithmetic wraps at the address-space index width. Do not
      // introduce nuw/nsw or assume nonnegative global-memory offsets.
      bytes = builder.CreateAdd(bytes, index, "shared.byte.offset");
    }
    Value *replacement = builder.CreateIntToPtr(bytes, pointer->getType(),
                                                "shared.load.address");
    replacements[pointer] = replacement;
    oldPointers.push_back(definition);
    use->set(replacement);
    ++changed;
  }
  // One pass only removes replaced roots; nested dead GEPs are harmless and
  // leaving them avoids invalidating pointers retained in the worklist.
  for (Instruction *pointer : oldPointers)
    if (pointer->use_empty())
      pointer->eraseFromParent();
  return changed;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  LLVMContext context;
  SMDiagnostic diagnostic;
  std::unique_ptr<Module> module = parseIRFile(Input, diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], errs());
    return 1;
  }
  if (verifyModule(*module, &errs()))
    return 2;
  if (Splits != 1 && LayoutShape.empty())
    return 5;
  if (GroupRows != 1 && LayoutShape.empty())
    return 5;
  if (!LayoutShape.empty() && (LayoutShape.size() != 3 || !MatmulPipeline))
    return 5;
  if (MatmulPipeline)
    optimizeMatmulPipeline(*module, LayoutShape.empty() ? 0 : LayoutShape[0],
                           LayoutShape.empty() ? 0 : LayoutShape[1],
                           LayoutShape.empty() ? 0 : LayoutShape[2], GroupRows,
                           Splits);
  unsigned changed = rewriteSharedLoads(*module);
  if (verifyModule(*module, &errs()))
    return 3;
  std::error_code error;
  raw_fd_ostream stream(Output, error, sys::fs::OF_Text);
  if (error) {
    errs() << error.message() << "\n";
    return 4;
  }
  module->print(stream, nullptr);
  errs() << "shared_load_uses_rewritten=" << changed << "\n";
  return 0;
}
