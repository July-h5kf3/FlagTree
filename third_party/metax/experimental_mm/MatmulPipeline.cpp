#include "MatmulPipeline.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

using namespace llvm;

namespace {

StringRef called(Value *value) {
  auto *call = dyn_cast<CallInst>(value);
  return call && call->getCalledFunction()
             ? call->getCalledFunction()->getName()
             : StringRef();
}

Value *field(Value *aggregate, unsigned index) {
  while (auto *insert = dyn_cast<InsertValueInst>(aggregate)) {
    if (insert->getNumIndices() != 1)
      return nullptr;
    if (insert->getIndices()[0] == index)
      return insert->getInsertedValueOperand();
    aggregate = insert->getAggregateOperand();
  }
  if (auto *constant = dyn_cast<Constant>(aggregate))
    return constant->getAggregateElement(index);
  return nullptr;
}

struct ByteOrigin {
  Value *value = nullptr;
  unsigned byte = 0;
  bool operator==(const ByteOrigin &other) const {
    return value == other.value && byte == other.byte;
  }
};

ByteOrigin byteOrigin(Value *value, unsigned byte, const DataLayout &layout,
                      unsigned depth = 0) {
  if (depth > 512)
    return {};
  if (auto *cast = dyn_cast<BitCastInst>(value))
    return byteOrigin(cast->getOperand(0), byte, layout, depth + 1);
  if (auto *extract = dyn_cast<ExtractValueInst>(value)) {
    if (extract->getNumIndices() != 1)
      return {};
    unsigned index = extract->getIndices()[0];
    if (Value *resolved = field(extract->getAggregateOperand(), index))
      return byteOrigin(resolved, byte, layout, depth + 1);
    auto *type =
        dyn_cast<StructType>(extract->getAggregateOperand()->getType());
    if (!type)
      return {};
    return {extract->getAggregateOperand(),
            unsigned(layout.getStructLayout(type)->getElementOffset(index)) +
                byte};
  }
  if (auto *extract = dyn_cast<ExtractElementInst>(value)) {
    auto *index = dyn_cast<ConstantInt>(extract->getIndexOperand());
    if (!index)
      return {};
    unsigned width = layout.getTypeStoreSize(extract->getType());
    return byteOrigin(extract->getVectorOperand(),
                      index->getZExtValue() * width + byte, layout, depth + 1);
  }
  if (auto *insert = dyn_cast<InsertElementInst>(value)) {
    auto *index = dyn_cast<ConstantInt>(insert->getOperand(2));
    if (!index)
      return {};
    unsigned width = layout.getTypeStoreSize(insert->getOperand(1)->getType());
    unsigned first = index->getZExtValue() * width;
    return byte >= first && byte < first + width
               ? byteOrigin(insert->getOperand(1), byte - first, layout,
                            depth + 1)
               : byteOrigin(insert->getOperand(0), byte, layout, depth + 1);
  }
  return {value, byte};
}

struct Packet {
  CallInst *load;
  StoreInst *store;
  unsigned base;
};

struct Chain {
  std::vector<CallInst *> mma;
  std::set<unsigned> leftLoads;
  std::set<unsigned> rightLoads;
  std::array<unsigned, 4> accumulatorFields;
  unsigned row = 0;
  unsigned column = 0;
};

struct Pattern {
  Function &function;
  const DataLayout &layout;
  BasicBlock *entry = nullptr;
  BasicBlock *header = nullptr;
  BasicBlock *body = nullptr;
  BasicBlock *exit = nullptr;
  PHINode *induction = nullptr;
  PHINode *accumulator = nullptr;
  CallInst *thread = nullptr;
  GlobalVariable *shared = nullptr;
  unsigned trips = 0;
  unsigned splits;
  bool allowMasked;
  std::vector<Packet> packets;
  std::vector<CallInst *> loads;
  std::vector<Chain> chains;
  std::vector<CallInst *> ordered;
  std::map<CallInst *, unsigned> chainIndex;
  std::map<CallInst *, std::array<ByteOrigin, 2>> operands;
  std::map<CallInst *, std::set<unsigned>> footprints;
  std::map<CallInst *, unsigned> retire;
  std::array<std::vector<CallInst *>, 4> groups;
  std::map<Value *, Value *> initial;
  std::string rejection;

  Pattern(Function &function, bool allowMasked = false, unsigned splits = 1)
      : function(function), layout(function.getParent()->getDataLayout()),
        splits(splits), allowMasked(allowMasked) {}

  bool require(bool condition, StringRef reason) {
    if (!condition && rejection.empty())
      rejection = reason.str();
    return condition;
  }

  std::optional<APInt> evaluate(Value *value, unsigned lane,
                                unsigned depth = 0) {
    if (depth > 256)
      return std::nullopt;
    if (value == shared)
      return APInt(32, 0);
    if (auto *constant = dyn_cast<ConstantInt>(value))
      return constant->getValue();
    if (value == thread)
      return APInt(32, lane);
    if (auto *gep = dyn_cast<GEPOperator>(value)) {
      auto base = evaluate(gep->getPointerOperand(), lane, depth + 1);
      if (!base || gep->getPointerAddressSpace() != 3)
        return std::nullopt;
      SmallMapVector<Value *, APInt, 4> terms;
      APInt constant(32, 0);
      if (!gep->collectOffset(layout, 32, terms, constant))
        return std::nullopt;
      APInt bytes = base->sextOrTrunc(32) + constant;
      for (auto &term : terms) {
        auto index = evaluate(term.first, lane, depth + 1);
        if (!index)
          return std::nullopt;
        bytes += index->sextOrTrunc(32) * term.second;
      }
      return bytes;
    }
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction)
      return std::nullopt;
    if (auto *cast = dyn_cast<CastInst>(instruction)) {
      auto operand = evaluate(cast->getOperand(0), lane, depth + 1);
      if (!operand || !cast->getType()->isIntegerTy())
        return std::nullopt;
      unsigned width = cast->getType()->getIntegerBitWidth();
      return cast->getOpcode() == Instruction::SExt
                 ? operand->sextOrTrunc(width)
                 : operand->zextOrTrunc(width);
    }
    if (auto *select = dyn_cast<SelectInst>(instruction)) {
      auto condition = evaluate(select->getCondition(), lane, depth + 1);
      if (!condition)
        return std::nullopt;
      return evaluate(condition->isZero() ? select->getFalseValue()
                                          : select->getTrueValue(),
                      lane, depth + 1);
    }
    if (instruction->getNumOperands() != 2)
      return std::nullopt;
    auto left = evaluate(instruction->getOperand(0), lane, depth + 1);
    auto right = evaluate(instruction->getOperand(1), lane, depth + 1);
    if (!left || !right)
      return std::nullopt;
    if (auto *comparison = dyn_cast<ICmpInst>(instruction))
      return APInt(
          1, ICmpInst::compare(*left, *right, comparison->getPredicate()));
    switch (instruction->getOpcode()) {
    case Instruction::Add:
      return *left + *right;
    case Instruction::Sub:
      return *left - *right;
    case Instruction::Mul:
      return *left * *right;
    case Instruction::And:
      return *left & *right;
    case Instruction::Or:
      return *left | *right;
    case Instruction::Xor:
      return *left ^ *right;
    case Instruction::Shl:
      return left->shl(*right);
    case Instruction::LShr:
      return left->lshr(*right);
    case Instruction::UDiv:
      if (!right->isZero())
        return left->udiv(*right);
      break;
    case Instruction::URem:
      if (!right->isZero())
        return left->urem(*right);
      break;
    default:
      break;
    }
    return std::nullopt;
  }

  bool matchLoop() {
    if (!require(function.getCallingConv() == 1023 &&
                     function.hasFnAttribute("metaxgpu-flat-work-group-size") &&
                     function.getFnAttribute("metaxgpu-flat-work-group-size")
                             .getValueAsString() == "1, 256",
                 "C550 kernel with 256 threads"))
      return false;
    if (!require(function.size() == 4,
                 "expected preheader, header, body, exit"))
      return false;
    entry = &function.getEntryBlock();
    auto *entryBranch = dyn_cast<BranchInst>(entry->getTerminator());
    if (!require(entryBranch && entryBranch->isUnconditional(), "entry branch"))
      return false;
    header = entryBranch->getSuccessor(0);
    auto *branch = dyn_cast<BranchInst>(header->getTerminator());
    if (!require(branch && branch->isConditional(), "loop condition"))
      return false;
    body = branch->getSuccessor(0);
    exit = branch->getSuccessor(1);
    auto *back = dyn_cast<BranchInst>(body->getTerminator());
    auto *comparison = dyn_cast<ICmpInst>(branch->getCondition());
    if (!require(back && back->isUnconditional() &&
                     back->getSuccessor(0) == header && comparison &&
                     comparison->getPredicate() == ICmpInst::ICMP_SLT,
                 "loop backedge or predicate"))
      return false;
    induction = dyn_cast<PHINode>(comparison->getOperand(0));
    auto *bound = dyn_cast<ConstantInt>(comparison->getOperand(1));
    if (!require(induction && induction->getType()->isIntegerTy(32) && bound &&
                     splits >= 1 && splits <= 64 &&
                     bound->getSExtValue() >= 2 &&
                     bound->getSExtValue() % splits == 0 &&
                     bound->getSExtValue() / splits >= 2 &&
                     bound->getSExtValue() / splits < 512,
                 "constant trip count"))
      return false;
    trips = bound->getZExtValue();
    for (Instruction &instruction : *header)
      if (!require(isa<PHINode>(instruction) || &instruction == comparison ||
                       &instruction == branch,
                   "unsupported loop header operation"))
        return false;
    for (PHINode &phi : header->phis()) {
      if (!require(phi.getNumIncomingValues() == 2 &&
                       phi.getBasicBlockIndex(entry) >= 0 &&
                       phi.getBasicBlockIndex(body) >= 0,
                   "loop phi shape"))
        return false;
      initial[&phi] = phi.getIncomingValueForBlock(entry);
      auto *type = dyn_cast<StructType>(phi.getType());
      if (type && type->getNumElements() == 64 &&
          llvm::all_of(type->elements(),
                       [](Type *type) { return type->isIntegerTy(32); }))
        accumulator = &phi;
    }
    if (!require(accumulator &&
                     isa<ConstantAggregateZero>(initial[accumulator]) &&
                     isa<ConstantInt>(initial[induction]) &&
                     cast<ConstantInt>(initial[induction])->isZero(),
                 "initial accumulators"))
      return false;
    auto *increment =
        dyn_cast<BinaryOperator>(induction->getIncomingValueForBlock(body));
    if (!require(increment && increment->getOpcode() == Instruction::Add &&
                     increment->getOperand(0) == induction &&
                     isa<ConstantInt>(increment->getOperand(1)) &&
                     cast<ConstantInt>(increment->getOperand(1))->isOne(),
                 "unit induction"))
      return false;
    for (Instruction &instruction : *entry)
      if (called(&instruction) == "llvm.mxc.thread.id.x")
        thread = cast<CallInst>(&instruction);
    for (GlobalVariable &global : function.getParent()->globals()) {
      if (global.getAddressSpace() != 3)
        continue;
      if (!require(!shared, "multiple shared allocations"))
        return false;
      shared = &global;
    }
    auto *sharedType =
        shared ? dyn_cast<ArrayType>(shared->getValueType()) : nullptr;
    if (!require(thread && sharedType && sharedType->getNumElements() == 0 &&
                     shared->isDeclaration(),
                 "dynamic shared ABI"))
      return false;
    unsigned barriers = 0;
    std::vector<StoreInst *> stores;
    std::vector<CallInst *> globalLoads;
    for (Instruction &instruction : *body) {
      StringRef name = called(&instruction);
      if (name == "llvm.mxc.ldg.predicator.v16i8")
        globalLoads.push_back(cast<CallInst>(&instruction));
      else if (name == "llvm.mxc.lds.v16i8")
        loads.push_back(cast<CallInst>(&instruction));
      else if (name == "llvm.mxc.mma.i32.16x16x16i8")
        ordered.push_back(cast<CallInst>(&instruction));
      else if (name == "llvm.mxc.barrier")
        ++barriers;
      else if (isa<CallInst>(instruction) && name != "llvm.mxc.icmp.i64.i32")
        return require(false, "unknown loop call");
      if (auto *fence = dyn_cast<FenceInst>(&instruction)) {
        bool release = fence->getOrdering() == AtomicOrdering::Release;
        bool acquire = fence->getOrdering() == AtomicOrdering::Acquire;
        Instruction *barrier =
            release ? fence->getNextNode() : fence->getPrevNode();
        if (!require(
                (release || acquire) && barrier &&
                    called(barrier) == "llvm.mxc.barrier" &&
                    fence->getSyncScopeID() ==
                        function.getContext().getOrInsertSyncScopeID("block"),
                "unrecognized loop fence"))
          return false;
      }
      if (!require(!instruction.mayReadOrWriteMemory() ||
                       isa<CallInst>(instruction) ||
                       isa<StoreInst>(instruction) ||
                       isa<FenceInst>(instruction),
                   "unknown loop memory operation"))
        return false;
      if (auto *store = dyn_cast<StoreInst>(&instruction)) {
        if (!require(store->getPointerAddressSpace() == 3 &&
                         !store->isVolatile() && !store->isAtomic() &&
                         layout.getTypeStoreSize(
                             store->getValueOperand()->getType()) == 16,
                     "non-packet loop store"))
          return false;
        stores.push_back(store);
      }
    }
    if (!require(globalLoads.size() == 16 && stores.size() == 16 &&
                     loads.size() == 32 && ordered.size() == 256 &&
                     barriers == 2,
                 "tile instruction signature"))
      return false;
    std::set<Value *> covered;
    for (StoreInst *store : stores) {
      auto origin = byteOrigin(store->getValueOperand(), 0, layout);
      if (!require(llvm::is_contained(globalLoads, origin.value) &&
                       origin.byte == 0,
                   "copy byte origin"))
        return false;
      for (unsigned byte = 0; byte < 16; ++byte)
        if (!require(byteOrigin(store->getValueOperand(), byte, layout) ==
                         ByteOrigin{origin.value, byte},
                     "copy byte permutation"))
          return false;
      auto *load = cast<CallInst>(origin.value);
      if (!require(load->arg_size() == 7 &&
                       isa<ConstantInt>(load->getArgOperand(1)) &&
                       cast<ConstantInt>(load->getArgOperand(1))->isZero(),
                   "global load immediate"))
        return false;
      for (unsigned flag = 3; flag < 7; ++flag)
        if (!require(isa<ConstantInt>(load->getArgOperand(flag)) &&
                         cast<ConstantInt>(load->getArgOperand(flag))
                                 ->getZExtValue() == (flag < 5),
                     "global load mode"))
          return false;
      auto *predicate = dyn_cast<CallInst>(load->getArgOperand(2));
      if (!require(
              allowMasked ||
                  (predicate && called(predicate) == "llvm.mxc.icmp.i64.i32" &&
                   isa<ConstantInt>(predicate->getArgOperand(0)) &&
                   cast<ConstantInt>(predicate->getArgOperand(0))->isOne() &&
                   isa<ConstantInt>(predicate->getArgOperand(1)) &&
                   cast<ConstantInt>(predicate->getArgOperand(1))->isZero() &&
                   isa<ConstantInt>(predicate->getArgOperand(2)) &&
                   cast<ConstantInt>(predicate->getArgOperand(2))
                           ->getZExtValue() == 34),
              "unmasked packet"))
        return false;
      auto base = evaluate(store->getPointerOperand(), 0);
      if (!require(base.has_value(), "shared packet base"))
        return false;
      unsigned start = base->getZExtValue();
      for (unsigned lane = 0; lane < 256; ++lane) {
        auto offset = evaluate(store->getPointerOperand(), lane);
        if (!require(offset && offset->getZExtValue() ==
                                   start + ((lane * 16) ^ (lane & 112)),
                     "packet lane swizzle"))
          return false;
      }
      covered.insert(load);
      packets.push_back({load, store, start});
    }
    std::sort(packets.begin(), packets.end(),
              [](const Packet &a, const Packet &b) { return a.base < b.base; });
    for (unsigned packet = 0; packet < 16; ++packet)
      if (!require(packets[packet].base == packet * 4096,
                   "shared packet coverage"))
        return false;
    if (!require(covered.size() == globalLoads.size(), "load/store bijection"))
      return false;
    for (CallInst *load : loads) {
      std::set<unsigned> packetSet;
      for (unsigned lane = 0; lane < 256; ++lane) {
        auto offset = evaluate(load->getArgOperand(0), lane);
        if (!require(offset && offset->getZExtValue() <= 65520 &&
                         offset->getZExtValue() % 16 == 0,
                     "LDS footprint"))
          return false;
        packetSet.insert(offset->getZExtValue() / 4096);
      }
      std::set<unsigned> groupSet;
      for (unsigned packet : packetSet)
        groupSet.insert((packet % 8) / 2);
      if (!require(groupSet.size() == 1, "LDS group footprint"))
        return false;
      groups[*groupSet.begin()].push_back(load);
      footprints[load] = packetSet;
    }
    return require(
        llvm::all_of(groups,
                     [](const auto &group) { return group.size() == 8; }),
        "four equal LDS groups");
  }

  bool matchChains() {
    for (CallInst *mma : ordered) {
      std::array<ByteOrigin, 2> pair;
      for (unsigned operand = 0; operand < 2; ++operand) {
        pair[operand] = byteOrigin(mma->getArgOperand(operand), 0, layout);
        if (!require(llvm::is_contained(loads, pair[operand].value) &&
                         pair[operand].byte % 4 == 0,
                     "MMA packed operand"))
          return false;
        for (unsigned byte = 0; byte < 4; ++byte)
          if (!require(byteOrigin(mma->getArgOperand(operand), byte, layout) ==
                           ByteOrigin{pair[operand].value,
                                      pair[operand].byte + byte},
                       "MMA byte order"))
            return false;
      }
      operands[mma] = pair;
      ByteOrigin parent = byteOrigin(mma->getArgOperand(2), 0, layout);
      unsigned chain;
      if (parent.value == accumulator) {
        chain = chains.size();
        chains.emplace_back();
        for (unsigned lane = 0; lane < 4; ++lane) {
          auto element = byteOrigin(mma->getArgOperand(2), lane * 4, layout);
          if (!require(element.value == accumulator && element.byte % 4 == 0 &&
                           element.byte < 256,
                       "initial MMA accumulator"))
            return false;
          for (unsigned byte = 0; byte < 4; ++byte)
            if (!require(byteOrigin(mma->getArgOperand(2), lane * 4 + byte,
                                    layout) ==
                             ByteOrigin{accumulator, element.byte + byte},
                         "initial accumulator byte order"))
              return false;
          chains.back().accumulatorFields[lane] = element.byte / 4;
        }
      } else {
        auto *previous = dyn_cast_or_null<CallInst>(parent.value);
        if (!require(previous && chainIndex.count(previous),
                     "accumulator dependency"))
          return false;
        chain = chainIndex[previous];
        if (!require(chains[chain].mma.back() == previous,
                     "accumulator branching"))
          return false;
        for (unsigned byte = 0; byte < 16; ++byte)
          if (!require(byteOrigin(mma->getArgOperand(2), byte, layout) ==
                           ByteOrigin{previous, byte},
                       "accumulator lane order"))
            return false;
      }
      chainIndex[mma] = chain;
      chains[chain].mma.push_back(mma);
      chains[chain].leftLoads.insert(llvm::find(loads, pair[0].value) -
                                     loads.begin());
      chains[chain].rightLoads.insert(llvm::find(loads, pair[1].value) -
                                      loads.begin());
    }
    if (!require(chains.size() == 16, "sixteen accumulators"))
      return false;
    std::set<std::set<unsigned>> rows, columns;
    std::set<unsigned> fields;
    for (Chain &chain : chains) {
      if (!require(chain.mma.size() == 16 && chain.leftLoads.size() == 4 &&
                       chain.rightLoads.size() == 4,
                   "MMA chain geometry"))
        return false;
      rows.insert(chain.leftLoads);
      columns.insert(chain.rightLoads);
      fields.insert(chain.accumulatorFields.begin(),
                    chain.accumulatorFields.end());
    }
    if (!require(rows.size() == 4 && columns.size() == 4 && fields.size() == 64,
                 "MMA tile geometry"))
      return false;
    std::map<CallInst *, std::tuple<unsigned, unsigned, unsigned, unsigned>>
        priorities;
    for (Chain &chain : chains) {
      chain.row = std::distance(rows.begin(), rows.find(chain.leftLoads));
      chain.column =
          std::distance(columns.begin(), columns.find(chain.rightLoads));
      for (unsigned depth = 0; depth < chain.mma.size(); ++depth) {
        unsigned a = chain.row, b = chain.column;
        if (std::max(a, b) < 3)
          priorities[chain.mma[depth]] = {std::max(a, b), a, b, depth};
        else {
          unsigned phase = ((a == 0 && b == 3) || (a == 3 && b == 0))   ? 3
                           : ((a == 1 && b == 3) || (a == 3 && b == 1)) ? 4
                                                                        : 5;
          priorities[chain.mma[depth]] = {phase, depth / 2, a, depth % 2};
        }
      }
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [&](CallInst *a, CallInst *b) {
                       return priorities[a] < priorities[b];
                     });
    for (unsigned position = 0; position < ordered.size(); ++position)
      for (auto operand : operands[ordered[position]])
        retire[cast<CallInst>(operand.value)] = position + 1;
    for (Chain &chain : chains) {
      std::vector<CallInst *> actual;
      for (CallInst *mma : ordered)
        if (chainIndex[mma] == chainIndex[chain.mma.front()])
          actual.push_back(mma);
      if (!require(actual == chain.mma, "scheduled accumulator order"))
        return false;
    }
    Value *backedge = accumulator->getIncomingValueForBlock(body);
    for (Chain &chain : chains)
      for (unsigned lane = 0; lane < 4; ++lane) {
        Value *scalar = field(backedge, chain.accumulatorFields[lane]);
        if (!require(scalar && byteOrigin(scalar, 0, layout) ==
                                   ByteOrigin{chain.mma.back(), lane * 4},
                     "accumulator backedge"))
          return false;
      }
    return true;
  }

  bool checkPointerEvolution() {
    std::set<std::pair<PHINode *, unsigned>> pointerFields;
    std::set<Value *> seen;
    std::function<bool(Value *)> visit = [&](Value *value) {
      if (!seen.insert(value).second)
        return true;
      if (isa<Constant>(value) || isa<Argument>(value))
        return true;
      if (auto *extract = dyn_cast<ExtractValueInst>(value)) {
        if (!require(extract->getNumIndices() == 1, "nested pointer extract"))
          return false;
        Value *aggregate = extract->getAggregateOperand();
        unsigned index = extract->getIndices()[0];
        if (auto *phi = dyn_cast<PHINode>(aggregate)) {
          if (!require(initial.count(phi) && extract->getType()->isPointerTy(),
                       "pointer phi extract"))
            return false;
          pointerFields.insert({phi, index});
          Value *first = field(initial[phi], index);
          return require(first, "initial pointer field") && visit(first);
        }
        if (Value *resolved = field(aggregate, index))
          return visit(resolved);
      }
      auto *instruction = dyn_cast<Instruction>(value);
      if (!require(instruction && !isa<PHINode>(instruction),
                   "unresolved address phi"))
        return false;
      StringRef name = called(instruction);
      if (isa<CallInst>(instruction))
        return require(name == "llvm.mxc.thread.id.x" ||
                           name == "llvm.mxc.block.id.x" ||
                           name == "llvm.mxc.icmp.i64.i32",
                       "address call side effect");
      if (!require(!instruction->mayReadOrWriteMemory() &&
                       !instruction->isTerminator(),
                   "address side effect"))
        return false;
      for (Value *operand : instruction->operands())
        if (!visit(operand))
          return false;
      return true;
    };
    for (Packet &packet : packets)
      if (!visit(packet.load->getArgOperand(0)))
        return false;
    if (!require(pointerFields.size() == 16,
                 "sixteen advancing packet pointers"))
      return false;
    for (auto [phi, index] : pointerFields) {
      Value *next = field(phi->getIncomingValueForBlock(body), index);
      auto *gep = dyn_cast_or_null<GetElementPtrInst>(next);
      if (!require(gep && gep->getSourceElementType()->isIntegerTy(8) &&
                       gep->getNumIndices() == 1,
                   "byte pointer increment"))
        return false;
      auto *step = dyn_cast<ConstantInt>(gep->getOperand(1));
      auto *base = dyn_cast<ExtractValueInst>(gep->getPointerOperand());
      if (!require(step && step->getSExtValue() == 256 && base &&
                       base->getAggregateOperand() == phi &&
                       base->getNumIndices() == 1 &&
                       base->getIndices()[0] == index,
                   "packet K stride"))
        return false;
    }
    for (BasicBlock *block : {header, body})
      for (Instruction &instruction : *block)
        for (User *user : instruction.users()) {
          auto *use = dyn_cast<Instruction>(user);
          if (!require(use && (use->getParent() == header ||
                               use->getParent() == body ||
                               (&instruction == accumulator &&
                                use->getParent() == exit)),
                       "external loop value"))
            return false;
        }
    return require(exit->phis().empty(), "exit phi unsupported");
  }
};

struct Schedule {
  Pattern &pattern;
  std::map<unsigned, std::vector<unsigned>> copyAt;
  std::map<unsigned, std::vector<CallInst *>> currentAt;
  std::map<unsigned, std::vector<CallInst *>> nextAt;
  std::set<unsigned> waitAt = {14, 84, 160, 216};
  std::array<unsigned, 16> copyOrder = {0, 1, 8,  9,  2, 3, 10, 11,
                                        4, 5, 12, 13, 6, 7, 14, 15};

  Schedule(Pattern &pattern) : pattern(pattern) {
    // C550 policy for a 128x128x256 tile. Legality is checked against the
    // matched graph and the outstanding-memory simulation, not the slot table.
    std::array<unsigned, 16> copies = {2,   8,   18,  28,  38,  48,  90,  100,
                                       110, 122, 162, 172, 182, 192, 222, 236};
    std::array<std::array<unsigned, 8>, 2> current = {
        std::array<unsigned, 8>{16, 24, 26, 34, 36, 44, 46, 54},
        std::array<unsigned, 8>{86, 88, 96, 98, 106, 108, 116, 128}};
    std::array<std::array<unsigned, 8>, 2> next = {
        std::array<unsigned, 8>{168, 170, 178, 180, 188, 190, 198, 200},
        std::array<unsigned, 8>{218, 220, 228, 230, 232, 234, 244, 248}};
    for (unsigned packet = 0; packet < copies.size(); ++packet)
      copyAt[copies[packet]].push_back(copyOrder[packet]);
    for (unsigned group = 0; group < 2; ++group) {
      for (unsigned index = 0; index < 8; ++index)
        currentAt[current[group][index]].push_back(
            pattern.groups[group + 2][index]);
      auto retained = pattern.groups[group];
      std::stable_sort(retained.begin(), retained.end(),
                       [&](CallInst *left, CallInst *right) {
                         return pattern.retire[left] < pattern.retire[right];
                       });
      for (unsigned index = 0; index < 8; ++index)
        nextAt[std::max(next[group][index], pattern.retire[retained[index]])]
            .push_back(retained[index]);
    }
  }

  bool validate() {
    std::array<int, 16> sharedVersion;
    sharedVersion.fill(-1);
    std::vector<std::pair<unsigned, unsigned>> pendingCopies;
    std::set<unsigned> pendingReads;
    std::map<CallInst *, unsigned> fragmentVersion;
    auto issueCopy = [&](unsigned packet, unsigned tile) {
      if (!pattern.require(!pendingReads.count(packet),
                           "copy overwrites outstanding LDS"))
        return false;
      for (auto item : pendingCopies)
        if (!pattern.require(item.first != packet,
                             "overlapping async copy destinations"))
          return false;
      pendingCopies.push_back({packet, tile});
      return true;
    };
    auto waitCopies = [&](unsigned pending) {
      unsigned complete =
          pendingCopies.size() > pending ? pendingCopies.size() - pending : 0;
      for (unsigned i = 0; i < complete; ++i)
        sharedVersion[pendingCopies[i].first] = pendingCopies[i].second;
      pendingCopies.erase(pendingCopies.begin(),
                          pendingCopies.begin() + complete);
    };
    auto issueRead = [&](CallInst *load, unsigned tile) {
      for (unsigned packet : pattern.footprints[load]) {
        if (!pattern.require(sharedVersion[packet] == int(tile),
                             "LDS before required copy completion"))
          return false;
        for (auto item : pendingCopies)
          if (!pattern.require(item.first != packet,
                               "LDS overlaps pending overwrite"))
            return false;
        pendingReads.insert(packet);
      }
      fragmentVersion[load] = tile;
      return true;
    };
    for (unsigned packet : copyOrder)
      if (!issueCopy(packet, 0))
        return false;
    waitCopies(12);
    for (CallInst *load : pattern.groups[0])
      if (!issueRead(load, 0))
        return false;
    waitCopies(8);
    pendingReads.clear();
    for (CallInst *load : pattern.groups[1])
      if (!issueRead(load, 0))
        return false;
    for (unsigned tile = 0; tile < pattern.trips; ++tile) {
      bool tail = tile + 1 == pattern.trips;
      for (unsigned position = 1; position <= pattern.ordered.size();
           ++position) {
        for (auto operand : pattern.operands[pattern.ordered[position - 1]]) {
          auto *load = cast<CallInst>(operand.value);
          if (!pattern.require(
                  fragmentVersion.count(load) && fragmentVersion[load] == tile,
                  "MMA before current fragment or after register overwrite"))
            return false;
        }
        if (tail) {
          if (position == 14)
            waitCopies(4);
          if (position == 84)
            waitCopies(0);
        } else {
          if (waitAt.count(position)) {
            waitCopies(6);
            pendingReads.clear();
          }
          for (unsigned packet : copyAt[position])
            if (!issueCopy(packet, tile + 1))
              return false;
        }
        for (CallInst *load : currentAt[position])
          if (!issueRead(load, tile))
            return false;
        if (!tail)
          for (CallInst *load : nextAt[position]) {
            if (!pattern.require(position >= pattern.retire[load],
                                 "next fragment before final consumer") ||
                !issueRead(load, tile + 1))
              return false;
          }
      }
    }
    return pattern.require(pendingCopies.empty(),
                           "outstanding copies after epilogue");
  }
};

#include "OutputLayout.inc"

struct Emitter {
  Pattern &pattern;
  Schedule &schedule;
  LLVMContext &context;
  Module &module;
  IRBuilder<> builder;
  Type *i32;
  Type *i8;
  FixedVectorType *fragmentType;
  FunctionCallee copyFunction;
  FunctionCallee arriveFunction;
  FunctionCallee barrierFunction;
  FunctionCallee mmaFunction;
  std::map<Value *, Value *> preValues;
  std::map<Value *, Value *> copyValues;
  std::map<CallInst *, Value *> sharedPointers;
  std::array<Value *, 16> globalPointers;
  std::array<Value *, 16> copySharedPointers;
  Value *copyMask = nullptr;
  OutputLayout *output;
  Value *splitStart = nullptr;
  std::array<Value *, 16> packetK;

  bool flattenGlobal;
  BasicBlock *tailBlock = nullptr;
  std::array<Value *, 16> globalRoots;
  std::array<Value *, 16> globalOffsets;

  Emitter(Pattern &pattern, Schedule &schedule, bool flattenGlobal,
          OutputLayout *output)
      : pattern(pattern), schedule(schedule),
        context(pattern.function.getContext()),
        module(*pattern.function.getParent()), builder(context), output(output),
        flattenGlobal(flattenGlobal) {
    i32 = builder.getInt32Ty();
    i8 = builder.getInt8Ty();
    fragmentType = FixedVectorType::get(i32, 4);
    auto *sharedPointerType = PointerType::get(context, 3);
    auto *globalPointerType = PointerType::get(context, 1);
    auto *i1 = builder.getInt1Ty();
    copyFunction = module.getOrInsertFunction(
        "llvm.mxc.ldg.predicator.bsm.v16i8",
        FunctionType::get(FixedVectorType::get(i8, 16),
                          {sharedPointerType, globalPointerType, i32,
                           builder.getInt64Ty(), i1, i1, i1, i1},
                          false));
    arriveFunction =
        module.getOrInsertFunction("llvm.mxc.arrive", builder.getVoidTy(), i32);
    barrierFunction = module.getOrInsertFunction("llvm.mxc.barrier.inst",
                                                 builder.getVoidTy());
    mmaFunction = pattern.ordered.front()->getCalledFunction();
  }

  Value *initialValue(Value *value, std::map<Value *, Value *> &cache) {
    auto found = cache.find(value);
    if (found != cache.end())
      return found->second;
    if (pattern.initial.count(value))
      return initialValue(pattern.initial[value], cache);
    if (isa<Constant>(value) || isa<Argument>(value))
      return value;
    if (auto *extract = dyn_cast<ExtractValueInst>(value)) {
      Value *aggregate = extract->getAggregateOperand();
      if (pattern.initial.count(aggregate))
        aggregate = pattern.initial[aggregate];
      if (Value *resolved = field(aggregate, extract->getIndices()[0]))
        return initialValue(resolved, cache);
    }
    auto *instruction = cast<Instruction>(value);
    if (called(instruction) == "llvm.mxc.block.id.x" ||
        called(instruction) == "llvm.mxc.thread.id.x" ||
        called(instruction) == "llvm.mxc.block.id.y")
      return instruction;
    Instruction *clone = instruction->clone();
    for (unsigned index = 0; index < clone->getNumOperands(); ++index)
      clone->setOperand(index,
                        initialValue(instruction->getOperand(index), cache));
    builder.Insert(clone, "mm.initial");
    cache[value] = clone;
    return clone;
  }

  void marker() {
    auto *type = FunctionType::get(builder.getVoidTy(), false);
    auto *assembly = InlineAsm::get(type, ";mm-order", "", true);
    auto *call = builder.CreateCall(assembly);
    call->setConvergent();
    call->setDoesNotThrow();
  }

  void wait(unsigned pending, bool reads) {
    builder.CreateCall(arriveFunction, {builder.getInt32(64 + pending)});
    if (reads)
      builder.CreateCall(arriveFunction, {builder.getInt32(4096)});
    builder.CreateCall(barrierFunction);
  }

  Value *read(CallInst *original, bool marked) {
    if (marked)
      marker();
    Value *result;
    if (marked) {
      result = builder.CreateAlignedLoad(fragmentType, sharedPointers[original],
                                         Align(16), "mm.fragment");
    } else {
      auto *bytes =
          builder.CreateCall(original->getCalledFunction(),
                             {sharedPointers[original]}, "mm.fragment.bytes");
      result = builder.CreateBitCast(bytes, fragmentType, "mm.fragment");
    }
    if (marked)
      marker();
    return result;
  }

  void copy(unsigned packet, Value *offset, bool marked) {
    Value *pointer = globalPointers[packet];
    if (flattenGlobal) {
      Value *bytes = offset ? builder.CreateAdd(globalOffsets[packet], offset,
                                                "mm.global.bytes")
                            : globalOffsets[packet];
      pointer =
          builder.CreateGEP(i8, globalRoots[packet], bytes, "mm.flat.global");
    } else if (offset) {
      pointer = builder.CreateGEP(i8, pointer, offset, "mm.next.global");
    }
    if (marked)
      marker();
    Value *mask = copyMask;
    if (pattern.splits > 1 && output->reduction % 256) {
      Value *index =
          offset ? builder.CreateAdd(packetK[packet], offset) : packetK[packet];
      Value *active = builder.CreateZExt(
          builder.CreateICmpULT(index, builder.getInt32(output->reduction)),
          i32);
      auto predicate = module.getOrInsertFunction(
          "llvm.mxc.icmp.i64.i32", builder.getInt64Ty(), i32, i32, i32);
      mask = builder.CreateCall(
          predicate, {active, builder.getInt32(0), builder.getInt32(34)});
    }
    builder.CreateCall(copyFunction, {copySharedPointers[packet], pointer,
                                      builder.getInt32(0), mask,
                                      builder.getTrue(), builder.getTrue(),
                                      builder.getFalse(), builder.getTrue()});
    if (marked)
      marker();
  }

  std::vector<Value *> emitTile(std::map<CallInst *, Value *> fragments,
                                std::vector<Value *> accumulators,
                                Value *offset,
                                std::map<CallInst *, Value *> &nextFragments) {
    bool tail = offset == nullptr;
    std::map<std::pair<CallInst *, unsigned>, Value *> packed;
    for (unsigned position = 1; position <= pattern.ordered.size();
         ++position) {
      CallInst *mma = pattern.ordered[position - 1];
      std::array<Value *, 2> operands;
      unsigned chain = pattern.chainIndex[mma];
      for (unsigned side = 0; side < 2; ++side) {
        auto origin = pattern.operands[mma][side];
        auto *load = cast<CallInst>(origin.value);
        unsigned lane = origin.byte / 4;
        auto key = std::make_pair(load, lane);
        if (!packed.count(key))
          packed[key] = builder.CreateExtractElement(
              fragments.at(load), builder.getInt32(lane), "mm.operand");
        operands[side] = packed[key];
      }
      accumulators[chain] = builder.CreateCall(
          mmaFunction, {operands[0], operands[1], accumulators[chain]},
          "mm.acc");
      if (tail) {
        if (position == 14)
          wait(4, false);
        if (position == 84)
          wait(0, false);
      } else {
        if (schedule.waitAt.count(position))
          wait(6, true);
        for (unsigned packet : schedule.copyAt[position])
          copy(packet, offset, true);
      }
      for (CallInst *load : schedule.currentAt[position])
        fragments[load] = read(load, true);
      if (!tail)
        for (CallInst *load : schedule.nextAt[position])
          nextFragments[load] = read(load, true);
    }
    return accumulators;
  }

  void run() {
    builder.SetInsertPoint(pattern.entry->getTerminator());
    Value *lane = pattern.thread;
    Value *swizzle =
        builder.CreateLShr(builder.CreateAnd(lane, builder.getInt32(112)), 4);
    copyValues[lane] = builder.CreateXor(lane, swizzle, "mm.copy.lane");
    Value *copyBytes = builder.CreateShl(lane, 4, "mm.copy.bytes");
    if (pattern.splits > 1)
      splitStart = builder.CreateMul(output->splitId,
                                     builder.getInt32(pattern.trips * 256),
                                     "mm.split.start");
    for (unsigned index = 0; index < pattern.packets.size(); ++index) {
      Packet &packet = pattern.packets[index];
      globalPointers[index] =
          initialValue(packet.load->getArgOperand(0), copyValues);
      if (flattenGlobal) {
        Value *pointer = globalPointers[index];
        Value *offset = builder.getInt32(0);
        while (auto *gep = dyn_cast<GetElementPtrInst>(pointer)) {
          Value *term = builder.CreateSExtOrTrunc(gep->getOperand(1), i32);
          uint64_t width =
              pattern.layout.getTypeAllocSize(gep->getSourceElementType());
          term = builder.CreateMul(term, builder.getInt32(width));
          offset = builder.CreateAdd(offset, term, "mm.initial.global.bytes");
          pointer = gep->getPointerOperand();
        }
        // OutputLayout proved every initial/future byte offset fits signed i32.
        globalRoots[index] = pointer;
        globalOffsets[index] =
            splitStart ? builder.CreateAdd(offset, splitStart) : offset;
        if (splitStart && output->reduction % 256) {
          const auto &coefficients = output->packetReduction.at(packet.load);
          Value *reductionIndex = builder.getInt32(coefficients[0]);
          for (unsigned bit = 0; bit < 8; ++bit) {
            if (!coefficients[bit + 1])
              continue;
            Value *laneBit = builder.CreateAnd(
                builder.CreateLShr(copyValues[lane], builder.getInt32(bit)),
                builder.getInt32(1));
            reductionIndex = builder.CreateAdd(
                reductionIndex,
                builder.CreateMul(laneBit,
                                  builder.getInt32(coefficients[bit + 1])));
          }
          packetK[index] =
              builder.CreateAdd(reductionIndex, splitStart, "mm.packet.k");
        }
      }
      Value *bytes =
          builder.CreateAdd(copyBytes, builder.getInt32(packet.base));
      copySharedPointers[index] =
          builder.CreateGEP(i8, pattern.shared, bytes, "mm.copy.shared");
    }
    copyMask =
        initialValue(pattern.packets.front().load->getArgOperand(2), preValues);
    for (CallInst *load : pattern.loads)
      sharedPointers[load] = initialValue(load->getArgOperand(0), preValues);
    for (unsigned packet : schedule.copyOrder)
      copy(packet, nullptr, false);
    std::map<CallInst *, Value *> retained;
    wait(12, false);
    for (CallInst *load : pattern.groups[0])
      retained[load] = read(load, false);
    wait(8, true);
    for (CallInst *load : pattern.groups[1])
      retained[load] = read(load, false);

    BasicBlock *header =
        BasicBlock::Create(context, "mm.loop", &pattern.function, pattern.exit);
    BasicBlock *body = BasicBlock::Create(context, "mm.steady",
                                          &pattern.function, pattern.exit);
    BasicBlock *tail =
        BasicBlock::Create(context, "mm.tail", &pattern.function, pattern.exit);
    tailBlock = tail;
    cast<BranchInst>(pattern.entry->getTerminator())->setSuccessor(0, header);
    builder.SetInsertPoint(header);
    auto *iteration = builder.CreatePHI(i32, 2, "mm.iteration");
    iteration->addIncoming(builder.getInt32(0), pattern.entry);
    std::vector<Value *> accumulators;
    for (unsigned chain = 0; chain < pattern.chains.size(); ++chain) {
      auto *phi = builder.CreatePHI(fragmentType, 2, "mm.acc.phi");
      phi->addIncoming(Constant::getNullValue(fragmentType), pattern.entry);
      accumulators.push_back(phi);
    }
    std::map<CallInst *, Value *> fragments;
    // Source instruction order keeps emission deterministic across executions.
    for (CallInst *load : pattern.loads) {
      if (!retained.count(load))
        continue;
      auto *phi = builder.CreatePHI(fragmentType, 2, "mm.fragment.phi");
      phi->addIncoming(retained[load], pattern.entry);
      fragments[load] = phi;
    }
    builder.CreateCondBr(
        builder.CreateICmpULT(iteration, builder.getInt32(pattern.trips - 1)),
        body, tail);
    builder.SetInsertPoint(body);
    Value *nextIteration =
        builder.CreateAdd(iteration, builder.getInt32(1), "mm.next.iteration");
    Value *offset = builder.CreateMul(nextIteration, builder.getInt32(256),
                                      "mm.next.offset");
    std::map<CallInst *, Value *> nextFragments;
    auto nextAccumulators =
        emitTile(fragments, accumulators, offset, nextFragments);
    for (unsigned chain = 0; chain < accumulators.size(); ++chain)
      cast<PHINode>(accumulators[chain])
          ->addIncoming(nextAccumulators[chain], body);
    for (auto &fragment : fragments)
      cast<PHINode>(fragment.second)
          ->addIncoming(nextFragments.at(fragment.first), body);
    iteration->addIncoming(nextIteration, body);
    builder.CreateBr(header);
    builder.SetInsertPoint(tail);
    auto finalAccumulators =
        emitTile(fragments, accumulators, nullptr, nextFragments);
    Value *aggregate = PoisonValue::get(pattern.accumulator->getType());
    for (unsigned chain = 0; chain < pattern.chains.size(); ++chain)
      for (unsigned lane = 0; lane < 4; ++lane) {
        Value *scalar = builder.CreateExtractElement(finalAccumulators[chain],
                                                     builder.getInt32(lane));
        aggregate = builder.CreateInsertValue(
            aggregate, scalar, pattern.chains[chain].accumulatorFields[lane]);
      }
    SmallVector<Use *> exitUses;
    for (Use &use : pattern.accumulator->uses())
      if (cast<Instruction>(use.getUser())->getParent() == pattern.exit)
        exitUses.push_back(&use);
    for (Use *use : exitUses)
      use->set(aggregate);
    builder.CreateBr(pattern.exit);
    pattern.header->dropAllReferences();
    pattern.body->dropAllReferences();
    pattern.header->eraseFromParent();
    pattern.body->eraseFromParent();
  }
};

} // namespace

unsigned optimizeMatmulPipeline(Module &module, unsigned rows, unsigned columns,
                                unsigned reduction, unsigned groupRows,
                                unsigned splits, bool rematerializeIndices) {
  unsigned matched = 0;
  for (Function &function : module) {
    if (function.isDeclaration())
      continue;
    Pattern pattern(function, rows != 0, splits);
    if (pattern.matchLoop() && pattern.matchChains() &&
        pattern.checkPointerEvolution()) {
      Schedule schedule(pattern);
      if (!schedule.validate()) {
        errs() << "skipped_mm=" << function.getName()
               << " reason=" << pattern.rejection << "\n";
        continue;
      }
      OutputLayout output(pattern, rows, columns, reduction, groupRows);
      if (rows &&
          !(output.collect() && output.proveOwnership() && output.plan())) {
        errs() << "skipped_layout=" << function.getName()
               << " reason=" << pattern.rejection << "\n";
        continue;
      }
      if (rows) {
        output.emit();
        errs() << "matched_layout=" << function.getName()
               << " owners=16384 stores=16\n";
      }
      ++matched;
      errs() << "matched_mm=" << function.getName()
             << " trips=" << pattern.trips
             << " packets=" << pattern.packets.size()
             << " chains=" << pattern.chains.size() << "\n";
      pattern.trips /= splits;
      Emitter emitter(pattern, schedule, rows != 0, &output);
      emitter.run();
      if (rows) {
        if (splits == 1)
          output.prefetchColumnScale(emitter.tailBlock);
        output.emitCtaOrder();
        if (rematerializeIndices)
          output.rematerializeTailIndices(emitter.tailBlock);
      }
    } else {
      errs() << "skipped_mm=" << function.getName()
             << " reason=" << pattern.rejection << "\n";
    }
  }
  return matched;
}
