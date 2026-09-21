// Included beside Pattern so ownership analysis shares the byte-origin rules.
struct TileAffine {
  int64_t row = 0, column = 0, constant = 0;
  bool valid = true;
  int64_t iteration = 0;
  TileAffine operator+(TileAffine other) const {
    TileAffine sum;
    sum.valid = valid && other.valid;
    sum.valid &=
        !__builtin_add_overflow(iteration, other.iteration, &sum.iteration);
    sum.valid &= !__builtin_add_overflow(row, other.row, &sum.row);
    sum.valid &= !__builtin_add_overflow(column, other.column, &sum.column);
    sum.valid &=
        !__builtin_add_overflow(constant, other.constant, &sum.constant);
    return sum;
  }
  TileAffine operator*(int64_t factor) const {
    TileAffine product;
    product.valid = valid;
    product.valid &=
        !__builtin_mul_overflow(iteration, factor, &product.iteration);
    product.valid &= !__builtin_mul_overflow(row, factor, &product.row);
    product.valid &= !__builtin_mul_overflow(column, factor, &product.column);
    product.valid &=
        !__builtin_mul_overflow(constant, factor, &product.constant);
    return product;
  }
  bool operator==(TileAffine other) const {
    return valid && other.valid && iteration == other.iteration &&
           row == other.row && column == other.column &&
           constant == other.constant;
  }
  bool isConstant() const { return row == 0 && column == 0 && iteration == 0; }
};

struct OutputLayout {
  Pattern &pattern;
  unsigned rows, columns, reduction, groupRows;
  unsigned rowScaleWidth;
  unsigned proofFirst = 0, proofLast = 0;
  Value *splitId = nullptr;
  Value *tracedAccumulator = nullptr;
  std::map<CallInst *, Value *> integerOutputs;
  std::map<CallInst *, std::array<int64_t, 9>> packetReduction;
  CallInst *block = nullptr;
  std::vector<CallInst *> conversions, globalStores;
  std::vector<StoreInst *> sharedStores;
  std::vector<GetElementPtrInst *> matrixRows[2], scaleRows[2];
  std::map<CallInst *, std::array<ByteOrigin, 2>> scales;
  std::map<std::pair<unsigned, Value *>, TileAffine> owners;
  std::map<std::pair<unsigned, Value *>, std::optional<TileAffine>> cache;
  std::vector<CallInst *> storeOrder;
  std::vector<CallInst *> columnLoads, rowLoads;
  CallInst *packedColumnScales = nullptr;
  std::array<int64_t, 8> laneCoefficients;
  std::map<CallInst *, int64_t> offsets;

  OutputLayout(Pattern &pattern, unsigned rows, unsigned columns,
               unsigned reduction, unsigned groupRows)
      : pattern(pattern), rows(rows), columns(columns), reduction(reduction),
        groupRows(groupRows), rowScaleWidth(rows % 4 == 0   ? 4
                                            : rows % 2 == 0 ? 2
                                                            : 1) {}

  bool require(bool condition, StringRef reason) {
    return pattern.require(condition, reason);
  }

  bool inRange(TileAffine value, unsigned width) {
    if (!value.valid)
      return false;
    __int128 low = value.constant, high = value.constant;
    for (auto term :
         {std::pair<int64_t, unsigned>{value.row, ((rows + 127) / 128) - 1},
          {value.column, columns / 128 - 1},
          {value.iteration, pattern.trips - 1}}) {
      __int128 extent = (__int128)term.first * term.second;
      low += std::min<__int128>(0, extent);
      high += std::max<__int128>(0, extent);
    }
    return low >= -((__int128)1 << (width - 1)) &&
           high < ((__int128)1 << (width - 1));
  }

  std::optional<TileAffine> evaluate(Value *value, unsigned lane,
                                     unsigned depth = 0) {
    if (depth > 512)
      return std::nullopt;
    auto key = std::make_pair(lane, value);
    auto found = cache.find(key);
    if (found != cache.end())
      return found->second;
    auto result = evaluateImpl(value, lane, depth);
    if (result) {
      if (rows <= 128)
        result->row = 0;
      if (columns == 128)
        result->column = 0;
    }
    if (result && value->getType()->isIntegerTy() &&
        !inRange(*result, value->getType()->getIntegerBitWidth()))
      result = std::nullopt;
    cache[key] = result;
    return result;
  }

  std::optional<TileAffine> evaluateImpl(Value *value, unsigned lane,
                                         unsigned depth) {
    if (auto *constant = dyn_cast<ConstantInt>(value))
      return TileAffine{0, 0, constant->getSExtValue()};
    if (value == pattern.induction)
      return TileAffine{0, 0, 0, true, 1};
    if (value == block && rows <= 128)
      return TileAffine{0, 1, 0};
    if (value == pattern.thread)
      return TileAffine{0, 0, int64_t(lane)};
    if (auto *cast = dyn_cast<CastInst>(value)) {
      auto operand = evaluate(cast->getOperand(0), lane, depth + 1);
      if (!operand || (cast->getOpcode() != Instruction::SExt &&
                       cast->getOpcode() != Instruction::ZExt))
        return std::nullopt;
      if (cast->getOpcode() == Instruction::ZExt && operand->constant < 0)
        return std::nullopt;
      return operand;
    }
    auto *operation = dyn_cast<BinaryOperator>(value);
    if (operation && operation->getOperand(0) == block &&
        (operation->getOpcode() == Instruction::SRem ||
         operation->getOpcode() == Instruction::URem ||
         operation->getOpcode() == Instruction::SDiv ||
         operation->getOpcode() == Instruction::UDiv)) {
      auto *divisor = dyn_cast<ConstantInt>(operation->getOperand(1));
      if (!divisor || divisor->getZExtValue() != ((rows + 127) / 128))
        return std::nullopt;
      if (operation->getOpcode() == Instruction::SRem ||
          operation->getOpcode() == Instruction::URem)
        return TileAffine{1, 0, 0};
      if (operation->getOpcode() == Instruction::SDiv ||
          operation->getOpcode() == Instruction::UDiv)
        return TileAffine{0, 1, 0};
      return std::nullopt;
    }
    if (!operation) {
      // All lane-only bit operations are evaluated with LLVM integer semantics.
      auto result = pattern.evaluate(value, lane);
      if (result)
        return TileAffine{0, 0, result->getSExtValue()};
      return std::nullopt;
    }
    auto left = evaluate(operation->getOperand(0), lane, depth + 1);
    auto right = evaluate(operation->getOperand(1), lane, depth + 1);
    if (!left || !right)
      return std::nullopt;
    if (left->isConstant() && right->isConstant()) {
      auto result = pattern.evaluate(value, lane);
      if (result)
        return TileAffine{0, 0, result->getSExtValue()};
    }
    switch (operation->getOpcode()) {
    case Instruction::Add:
      return *left + *right;
    case Instruction::Sub:
      return *left + *right * -1;
    case Instruction::Mul:
      if (right->isConstant() && right->constant >= -(1LL << 24) &&
          right->constant <= (1LL << 24))
        return *left * right->constant;
      if (left->isConstant() && left->constant >= -(1LL << 24) &&
          left->constant <= (1LL << 24))
        return *right * left->constant;
      break;
    default:
      break;
    }
    return std::nullopt;
  }

  std::optional<TileAffine> pointerOffset(Value *pointer, Value *root,
                                          unsigned lane) {
    if (pointer == root)
      return TileAffine{};
    if (pattern.initial.count(pointer))
      return pointerOffset(pattern.initial.at(pointer), root, lane);
    if (auto *extract = dyn_cast<ExtractValueInst>(pointer)) {
      Value *aggregate = extract->getAggregateOperand();
      if (pattern.initial.count(aggregate))
        aggregate = pattern.initial.at(aggregate);
      Value *resolved = field(aggregate, extract->getIndices()[0]);
      if (!resolved)
        return std::nullopt;
      return pointerOffset(resolved, root, lane);
    }
    auto *gep = dyn_cast<GEPOperator>(pointer);
    if (!gep || gep->getNumIndices() != 1)
      return std::nullopt;
    auto base = pointerOffset(gep->getPointerOperand(), root, lane);
    auto index = evaluate(gep->getOperand(1), lane);
    if (!base || !index || !gep->getSourceElementType()->isSized())
      return std::nullopt;
    TileAffine address = *base + *index * pattern.layout.getTypeAllocSize(
                                              gep->getSourceElementType());
    if (!address.valid)
      return std::nullopt;
    return address;
  }

  bool truePredicate(Value *value) {
    auto *call = dyn_cast<CallInst>(value);
    if (!call || called(call) != "llvm.mxc.icmp.i64.i32" ||
        call->arg_size() != 3)
      return false;
    unsigned expected[] = {1, 0, 34};
    for (unsigned i = 0; i < 3; ++i) {
      auto *constant = dyn_cast<ConstantInt>(call->getArgOperand(i));
      if (!constant || constant->getZExtValue() != expected[i])
        return false;
    }
    return true;
  }

  std::optional<bool> truth(Value *value, unsigned lane, unsigned rowTile,
                            unsigned depth = 0) {
    if (depth > 128)
      return std::nullopt;
    if (auto *constant = dyn_cast<ConstantInt>(value)) {
      if (constant->isZero())
        return false;
      if (constant->isOne())
        return true;
      return std::nullopt;
    }
    if (auto *cast = dyn_cast<ZExtInst>(value)) {
      if (!cast->getOperand(0)->getType()->isIntegerTy(1))
        return std::nullopt;
      return truth(cast->getOperand(0), lane, rowTile, depth + 1);
    }
    if (auto *operation = dyn_cast<BinaryOperator>(value)) {
      if (!operation->getType()->isIntegerTy(1))
        return std::nullopt;
      auto left = truth(operation->getOperand(0), lane, rowTile, depth + 1);
      auto right = truth(operation->getOperand(1), lane, rowTile, depth + 1);
      if (operation->getOpcode() == Instruction::And &&
          ((left && !*left) || (right && !*right)))
        return false;
      if (operation->getOpcode() == Instruction::Or &&
          ((left && *left) || (right && *right)))
        return true;
      if (!left || !right)
        return std::nullopt;
      if (operation->getOpcode() == Instruction::And)
        return *left && *right;
      if (operation->getOpcode() == Instruction::Or)
        return *left || *right;
      return std::nullopt;
    }
    auto *compare = dyn_cast<ICmpInst>(value);
    if (!compare)
      return std::nullopt;
    auto left = evaluate(compare->getOperand(0), lane);
    auto right = evaluate(compare->getOperand(1), lane);
    if (!left || !right)
      return std::nullopt;
    auto bounds = [&](TileAffine expression) {
      __int128 base = (__int128)expression.row * rowTile + expression.constant +
                      (__int128)expression.iteration * proofFirst;
      __int128 extent = (__int128)expression.column * (columns / 128 - 1);
      __int128 reductionExtent =
          (__int128)expression.iteration * (proofLast - proofFirst);
      return std::pair<__int128, __int128>{
          base + std::min<__int128>(0, extent) +
              std::min<__int128>(0, reductionExtent),
          base + std::max<__int128>(0, extent) +
              std::max<__int128>(0, reductionExtent)};
    };
    if (compare->isUnsigned() &&
        (bounds(*left).first < 0 || bounds(*right).first < 0))
      return std::nullopt;
    TileAffine difference = *left + *right * -1;
    if (!difference.valid)
      return std::nullopt;
    auto [low, high] = bounds(difference);
    switch (compare->getPredicate()) {
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_ULT:
      if (high < 0)
        return true;
      if (low >= 0)
        return false;
      break;
    case ICmpInst::ICMP_SLE:
    case ICmpInst::ICMP_ULE:
      if (high <= 0)
        return true;
      if (low > 0)
        return false;
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_UGT:
      if (low > 0)
        return true;
      if (high <= 0)
        return false;
      break;
    case ICmpInst::ICMP_SGE:
    case ICmpInst::ICMP_UGE:
      if (low >= 0)
        return true;
      if (high < 0)
        return false;
      break;
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_NE:
      if (low == 0 && high == 0)
        return compare->getPredicate() == ICmpInst::ICMP_EQ;
      if (low > 0 || high < 0)
        return compare->getPredicate() == ICmpInst::ICMP_NE;
      break;
    default:
      break;
    }
    return std::nullopt;
  }

  std::optional<bool> predicate(Value *value, unsigned lane, unsigned rowTile) {
    auto *call = dyn_cast<CallInst>(value);
    if (!call || called(call) != "llvm.mxc.icmp.i64.i32" ||
        call->arg_size() != 3)
      return std::nullopt;
    auto *zero = dyn_cast<ConstantInt>(call->getArgOperand(1));
    auto *comparison = dyn_cast<ConstantInt>(call->getArgOperand(2));
    if (!zero || !zero->isZero() || !comparison ||
        (comparison->getZExtValue() != 33 && comparison->getZExtValue() != 34))
      return std::nullopt;
    return truth(call->getArgOperand(0), lane, rowTile);
  }

  bool traceScales(Value *value, unsigned byte, std::vector<ByteOrigin> &loads,
                   unsigned &accumulators, unsigned depth = 0) {
    if (depth > 32)
      return false;
    ByteOrigin origin = byteOrigin(value, byte, pattern.layout);
    if (!origin.value || origin.byte % 4)
      return false;
    if (auto *cast = dyn_cast<SIToFPInst>(origin.value)) {
      ByteOrigin integer = byteOrigin(cast->getOperand(0), 0, pattern.layout);
      if (origin.byte || integer.value != pattern.accumulator)
        return false;
      tracedAccumulator = cast->getOperand(0);
      ++accumulators;
      return true;
    }
    auto *call = dyn_cast<CallInst>(origin.value);
    if (!call)
      return false;
    if (called(call) == "llvm.mxc.ldg.predicator.v4f32" ||
        called(call) == "llvm.mxc.ldg.predicator.v2f32" ||
        called(call) == "llvm.mxc.ldg.predicator.f32") {
      if (origin.byte >= pattern.layout.getTypeStoreSize(call->getType()))
        return false;
      loads.push_back(origin);
      return true;
    }
    if (called(call) != "llvm.mxc.pk.fma.f32" || origin.byte >= 8)
      return false;
    auto *addend = dyn_cast<Constant>(call->getArgOperand(2));
    return addend && addend->isNullValue() &&
           traceScales(call->getArgOperand(0), origin.byte, loads, accumulators,
                       depth + 1) &&
           traceScales(call->getArgOperand(1), origin.byte, loads, accumulators,
                       depth + 1);
  }

  bool collect() {
    if (!require(
            rows && columns && rows < (1u << 31) - 127 && columns % 128 == 0 &&
                uint64_t((rows + 127) / 128) * 128 * columns < (1ULL << 31) &&
                reduction > 0 && reduction % 16 == 0 &&
                (uint64_t(reduction) + 255) / 256 == pattern.trips &&
                (pattern.splits > 1 || reduction % 256 == 0) &&
                uint64_t(rows) * columns * pattern.splits < (1ULL << 31) &&
                uint64_t((rows + 127) / 128) * 128 * reduction < (1ULL << 31) &&
                uint64_t(columns) * reduction < (1ULL << 31),
            "layout specialization shape/range"))
      return false;
    if (!require(pattern.function.arg_size() >= 5, "layout pointer ABI"))
      return false;
    for (unsigned i = 0; i < 5; ++i)
      if (!require(pattern.function.getArg(i)->getType()->isPointerTy() &&
                       pattern.function.getArg(i)
                               ->getType()
                               ->getPointerAddressSpace() == 1,
                   "layout global ABI"))
        return false;
    for (Instruction &instruction : *pattern.entry) {
      if (!require(!instruction.mayWriteToMemory(), "entry memory write"))
        return false;
      if (called(&instruction) == "llvm.mxc.block.id.x") {
        if (!require(!block, "one CTA identifier"))
          return false;
        block = cast<CallInst>(&instruction);
      }
      auto *gep = dyn_cast<GetElementPtrInst>(&instruction);
      if (!gep)
        continue;
      for (unsigned operand = 0; operand < 2; ++operand)
        if (gep->getPointerOperand() == pattern.function.getArg(operand))
          matrixRows[operand].push_back(gep);
    }
    if (!require(block && matrixRows[0].size() == 8 &&
                     matrixRows[1].size() == 8,
                 "matrix row address roots"))
      return false;
    for (Instruction &instruction : *pattern.exit) {
      auto name = called(&instruction);
      if (name == "llvm.mxc.cvt.f32tobf16")
        conversions.push_back(cast<CallInst>(&instruction));
      if (name == "llvm.mxc.stg.predicator.v8i16")
        globalStores.push_back(cast<CallInst>(&instruction));
      if (auto *store = dyn_cast<StoreInst>(&instruction))
        sharedStores.push_back(store);
      if (auto *gep = dyn_cast<GetElementPtrInst>(&instruction))
        for (unsigned operand = 0; operand < 2; ++operand)
          if (gep->getPointerOperand() == pattern.function.getArg(operand + 2))
            scaleRows[operand].push_back(gep);
    }
    if (!require(conversions.size() == 64 && globalStores.size() == 8 &&
                     sharedStores.size() == 16 &&
                     scaleRows[0].size() == 16 / rowScaleWidth &&
                     scaleRows[1].size() == 4,
                 "output tile geometry"))
      return false;
    unsigned phase = 0, barriers = 0, scaleLoads = 0;
    for (Instruction &instruction : *pattern.exit) {
      StringRef name = called(&instruction);
      if (llvm::is_contained(conversions, &instruction)) {
        if (!require(phase == 0, "conversion before shared exchange"))
          return false;
      } else if (name == "llvm.mxc.barrier") {
        ++barriers;
        ++phase;
      } else if (isa<StoreInst>(instruction)) {
        if (!require(phase == 1, "shared stores between output barriers"))
          return false;
      } else if (isa<LoadInst>(instruction) ||
                 name == "llvm.mxc.stg.predicator.v8i16") {
        if (!require(phase == 2, "output loads after exchange barrier"))
          return false;
      } else if (name == "llvm.mxc.ldg.predicator.v4f32" ||
                 name == "llvm.mxc.ldg.predicator.v2f32" ||
                 name == "llvm.mxc.ldg.predicator.f32") {
        auto *load = cast<CallInst>(&instruction);
        auto *pointer = dyn_cast<GetElementPtrInst>(load->getArgOperand(0));
        bool row = pointer && llvm::is_contained(scaleRows[0], pointer);
        bool column = pointer && llvm::is_contained(scaleRows[1], pointer);
        if (!require(phase == 0 &&
                         ((row && pattern.layout.getTypeStoreSize(
                                      load->getType()) == 4 * rowScaleWidth) ||
                          (column && name == "llvm.mxc.ldg.predicator.f32")),
                     "scale load geometry"))
          return false;
        for (unsigned position : {1u, 3u, 4u, 5u, 6u}) {
          auto *flag = dyn_cast<ConstantInt>(load->getArgOperand(position));
          unsigned expected = position == 3 || position == 4;
          if (!require(flag && flag->getZExtValue() == expected,
                       "scale load mode"))
            return false;
        }
        if (column)
          columnLoads.push_back(load);
        else
          rowLoads.push_back(load);
        ++scaleLoads;
      } else if (instruction.mayHaveSideEffects() &&
                 !isa<FenceInst>(instruction) &&
                 !isa<ReturnInst>(instruction)) {
        return require(false, "unknown output side effect");
      }
    }
    if (!require(barriers == 2 && scaleLoads == 16 / rowScaleWidth + 4,
                 "output barrier and scale-load count"))
      return false;
    for (CallInst *store : globalStores) {
      unsigned positions[] = {1, 4, 5, 6};
      unsigned expected[] = {0, 1, 0, 0};
      for (unsigned i = 0; i < 4; ++i) {
        auto *flag = dyn_cast<ConstantInt>(store->getArgOperand(positions[i]));
        if (!require(flag && flag->getZExtValue() == expected[i],
                     "output store mode"))
          return false;
      }
    }
    for (CallInst *conversion : conversions) {
      std::vector<ByteOrigin> loads;
      unsigned accumulators = 0;
      if (!require(traceScales(conversion->getArgOperand(0), 0, loads,
                               accumulators) &&
                       loads.size() == 2 && accumulators == 1,
                   "output scale arithmetic"))
        return false;
      integerOutputs[conversion] = tracedAccumulator;
      std::array<ByteOrigin, 2> ordered{};
      for (ByteOrigin origin : loads) {
        auto *pointer = dyn_cast<GetElementPtrInst>(
            cast<CallInst>(origin.value)->getArgOperand(0));
        if (!require(pointer, "scale pointer GEP"))
          return false;
        int operand =
            pointer->getPointerOperand() == pattern.function.getArg(2)   ? 0
            : pointer->getPointerOperand() == pattern.function.getArg(3) ? 1
                                                                         : -1;
        if (!require(operand >= 0 && !ordered[operand].value,
                     "row and column scale identity"))
          return false;
        ordered[operand] = origin;
      }
      scales[conversion] = ordered;
    }
    return true;
  }

  bool proveOwnership() {
    using Token = std::pair<unsigned, Value *>;
    std::map<unsigned, Token> shared;
    for (unsigned lane = 0; lane < 256; ++lane) {
      for (StoreInst *store : sharedStores) {
        auto address = pattern.evaluate(store->getPointerOperand(), lane);
        if (!require(
                address && !store->isVolatile() && !store->isAtomic() &&
                    store->getValueOperand()->getType() ==
                        FixedVectorType::get(
                            Type::getInt16Ty(pattern.function.getContext()), 4),
                "shared exchange store"))
          return false;
        for (unsigned element = 0; element < 4; ++element) {
          ByteOrigin origin =
              byteOrigin(store->getValueOperand(), element * 2, pattern.layout);
          if (!require(origin.byte == 0 &&
                           llvm::is_contained(conversions, origin.value) &&
                           byteOrigin(store->getValueOperand(), element * 2 + 1,
                                      pattern.layout) ==
                               ByteOrigin{origin.value, 1},
                       "BF16 shared byte ownership"))
            return false;
          if (!require(shared
                           .emplace(address->getZExtValue() + element * 2,
                                    Token{lane, origin.value})
                           .second,
                       "unique shared output producer"))
            return false;
        }
      }
    }
    std::set<int64_t> destinations;
    for (unsigned lane = 0; lane < 256; ++lane) {
      for (CallInst *store : globalStores) {
        auto destination = pointerOffset(store->getArgOperand(0),
                                         pattern.function.getArg(4), lane);
        if (!require(destination &&
                         destination->row ==
                             (rows <= 128 ? 0 : int64_t(256) * columns) &&
                         destination->column == (columns == 128 ? 0 : 256),
                     "affine global output address/predicate"))
          return false;
        for (unsigned rowTile = 0; rowTile < (rows + 127) / 128; ++rowTile) {
          auto active = predicate(store->getArgOperand(3), lane, rowTile);
          bool expected =
              rowTile * 128 + destination->constant / (2 * columns) < rows;
          if (!require(active && *active == expected,
                       "affine global output address/predicate"))
            return false;
        }
        for (unsigned element = 0; element < 8; ++element) {
          ByteOrigin origin =
              byteOrigin(store->getArgOperand(2), element * 2, pattern.layout);
          auto *load = dyn_cast_or_null<LoadInst>(origin.value);
          if (!require(load && load->getPointerAddressSpace() == 3 &&
                           !load->isVolatile() && !load->isAtomic() &&
                           origin.byte % 2 == 0 &&
                           origin.byte < pattern.layout.getTypeStoreSize(
                                             load->getType()) &&
                           byteOrigin(store->getArgOperand(2), element * 2 + 1,
                                      pattern.layout) ==
                               ByteOrigin{origin.value, origin.byte + 1},
                       "shared exchange load"))
            return false;
          auto address = pattern.evaluate(load->getPointerOperand(), lane);
          if (!require(bool(address), "shared output address"))
            return false;
          auto token = shared.find(address->getZExtValue() + origin.byte);
          if (!require(token != shared.end(), "initialized shared output"))
            return false;
          TileAffine absolute =
              *destination + TileAffine{0, 0, int64_t(2 * element)};
          if (!require(owners.emplace(token->second, absolute).second &&
                           destinations.insert(absolute.constant).second,
                       "unique output ownership"))
            return false;
        }
      }
    }
    for (unsigned lane = 0; lane < 256; ++lane)
      for (CallInst *conversion : conversions) {
        auto found = owners.find({lane, conversion});
        if (!require(found != owners.end(), "complete output ownership"))
          return false;
        TileAffine coordinates[2];
        for (unsigned operand = 0; operand < 2; ++operand) {
          auto origin = scales[conversion][operand];
          auto index =
              pointerOffset(cast<CallInst>(origin.value)->getArgOperand(0),
                            pattern.function.getArg(operand + 2), lane);
          if (!require(index && index->constant % 4 == 0 &&
                           index->row % 4 == 0 && index->column % 4 == 0,
                       "scale alignment"))
            return false;
          coordinates[operand] = {index->row / 4, index->column / 4,
                                  (index->constant + origin.byte) / 4};
          auto expected = operand == 0
                              ? TileAffine{rows <= 128 ? 0 : 128, 0,
                                           coordinates[operand].constant}
                              : TileAffine{0, columns == 128 ? 0 : 128,
                                           coordinates[operand].constant};
          if (!require(coordinates[operand] == expected &&
                           coordinates[operand].constant >= 0 &&
                           coordinates[operand].constant < 128,
                       "scale tile coordinates"))
            return false;
        }
        if (!require(found->second ==
                         (coordinates[0] * columns + coordinates[1]) * 2,
                     "scale/output coordinate agreement"))
          return false;
      }
    return require(owners.size() == 16384 && destinations.size() == 16384,
                   "complete output tile");
  }

  static unsigned permute(unsigned index, unsigned operand) {
    return operand == 0
               ? ((index & 3) | ((index & 96) >> 3) | ((index & 28) << 2))
               : (((index & 31) << 2) | ((index & 96) >> 5));
  }

  bool plan() {
    if (!require(groupRows >= 1 && groupRows <= 64, "CTA group size"))
      return false;
    unsigned rowTiles = ((rows + 127) / 128), columnTiles = columns / 128;
    std::set<unsigned> mappedCtas;
    for (unsigned cta = 0; cta < rowTiles * columnTiles; ++cta) {
      unsigned first = cta / (groupRows * columnTiles) * groupRows;
      unsigned count = std::min(rowTiles - first, groupRows);
      unsigned within = cta % (groupRows * columnTiles);
      unsigned mapped = first + within % count + (within / count) * rowTiles;
      if (!require(mapped < rowTiles * columnTiles &&
                       mappedCtas.insert(mapped).second,
                   "CTA order bijection"))
        return false;
    }
    std::set<int64_t> inputBytes[2];
    for (Packet packet : pattern.packets) {
      for (unsigned lane = 0; lane < 256; ++lane) {
        auto a = pointerOffset(packet.load->getArgOperand(0),
                               pattern.function.getArg(0), lane);
        auto b = pointerOffset(packet.load->getArgOperand(0),
                               pattern.function.getArg(1), lane);
        if (!require(bool(a) != bool(b), "input packet matrix identity"))
          return false;
        unsigned operand = a ? 0 : 1;
        TileAffine address = a ? *a : *b;
        TileAffine expected =
            operand == 0
                ? TileAffine{rows <= 128 ? 0 : int64_t(128) * reduction, 0,
                             address.constant}
                : TileAffine{0, columns == 128 ? 0 : int64_t(128) * reduction,
                             address.constant};
        if (!require(address == expected && address.constant >= 0 &&
                         address.constant / reduction < 128 &&
                         address.constant % reduction < 256 &&
                         address.constant % 16 == 0,
                     "input packet tile coordinates"))
          return false;
        for (auto interval :
             {std::pair<unsigned, unsigned>{0, pattern.trips - 2},
              {pattern.trips - 1, pattern.trips - 1}}) {
          proofFirst = interval.first;
          proofLast = interval.second;
          for (unsigned rowTile = 0; rowTile < (rows + 127) / 128; ++rowTile) {
            auto active =
                predicate(packet.load->getArgOperand(2), lane, rowTile);
            bool expected =
                (operand == 1 ||
                 rowTile * 128 + address.constant / reduction < rows) &&
                address.constant % reduction + uint64_t(proofLast) * 256 <
                    reduction;
            if (!require(active && *active == expected,
                         "input packet row/reduction mask"))
              return false;
          }
        }
        proofFirst = proofLast = 0;
        auto &coefficients = packetReduction[packet.load];
        int64_t laneReduction = address.constant % reduction;
        if (lane == 0)
          coefficients[0] = laneReduction;
        for (unsigned bit = 0; bit < 8; ++bit)
          if (lane == (1u << bit))
            coefficients[bit + 1] = laneReduction - coefficients[0];
        for (unsigned byte = 0; byte < 16; ++byte)
          if (!require(
                  inputBytes[operand].insert(address.constant + byte).second,
                  "unique input packet bytes"))
            return false;
      }
    }
    for (Packet packet : pattern.packets) {
      const auto &coefficients = packetReduction.at(packet.load);
      for (unsigned lane = 0; lane < 256; ++lane) {
        auto address = pointerOffset(packet.load->getArgOperand(0),
                                     pattern.function.getArg(0), lane);
        if (!address)
          address = pointerOffset(packet.load->getArgOperand(0),
                                  pattern.function.getArg(1), lane);
        int64_t expected = coefficients[0];
        for (unsigned bit = 0; bit < 8; ++bit)
          if (lane & (1u << bit))
            expected += coefficients[bit + 1];
        if (!require(address && expected == address->constant % reduction,
                     "packet reduction lane formula"))
          return false;
      }
    }
    if (!require(inputBytes[0].size() == 32768 && inputBytes[1].size() == 32768,
                 "complete input tile coverage"))
      return false;
    for (unsigned operand = 0; operand < 2; ++operand) {
      std::set<unsigned> permutation;
      for (unsigned index = 0; index < 128; ++index)
        permutation.insert(permute(index, operand));
      if (!require(permutation.size() == 128, "bijective tile permutation"))
        return false;
      for (GetElementPtrInst *gep : matrixRows[operand]) {
        if (!require(gep->getNumIndices() == 1, "matrix row GEP arity"))
          return false;
        auto *multiply = dyn_cast<BinaryOperator>(gep->getOperand(1));
        auto *stride =
            multiply ? dyn_cast<ConstantInt>(multiply->getOperand(1)) : nullptr;
        if (!require(gep->getNumIndices() == 1 &&
                         gep->getSourceElementType()->isIntegerTy(8) &&
                         multiply &&
                         multiply->getOpcode() == Instruction::Mul && stride &&
                         stride->getZExtValue() == reduction,
                     "matrix row stride"))
          return false;
        for (unsigned lane = 0; lane < 256; ++lane) {
          auto row = evaluate(multiply->getOperand(0), lane);
          TileAffine expected = operand == 0
                                    ? TileAffine{rows <= 128 ? 0 : 128, 0,
                                                 row ? row->constant : 0}
                                    : TileAffine{0, columns == 128 ? 0 : 128,
                                                 row ? row->constant : 0};
          if (!require(row && *row == expected && row->constant >= 0 &&
                           row->constant < 128,
                       "matrix row tile coordinate"))
            return false;
        }
      }
      for (GetElementPtrInst *gep : scaleRows[operand]) {
        if (!require(gep->getNumIndices() == 1 &&
                         gep->getSourceElementType()->isFloatTy(),
                     "scale GEP type"))
          return false;
        for (unsigned lane = 0; lane < 256; ++lane) {
          auto index = evaluate(gep->getOperand(1), lane);
          if (!require(index && (operand == 1 ||
                                 (index->constant % rowScaleWidth == 0 &&
                                  permute(index->constant + rowScaleWidth - 1,
                                          operand) ==
                                      permute(index->constant, operand) +
                                          rowScaleWidth - 1)),
                       "row-scale vector contiguity"))
            return false;
        }
      }
    }
    for (unsigned operand = 0; operand < 2; ++operand) {
      const auto &loads = operand == 0 ? rowLoads : columnLoads;
      for (CallInst *load : loads) {
        auto *gep = cast<GetElementPtrInst>(load->getArgOperand(0));
        for (unsigned lane = 0; lane < 256; ++lane) {
          auto index = evaluate(gep->getOperand(1), lane);
          if (!require(bool(index), "scale mask index"))
            return false;
          for (unsigned rowTile = 0; rowTile < (rows + 127) / 128; ++rowTile) {
            auto active = predicate(load->getArgOperand(2), lane, rowTile);
            bool expected =
                operand == 1 || rowTile * 128 + index->constant < rows;
            if (!require(active && *active == expected, "scale row mask"))
              return false;
          }
        }
      }
    }
    if (!require(columnLoads.size() == 4, "four column scale loads"))
      return false;
    for (unsigned lane = 0; lane < 256; ++lane) {
      unsigned first = 0;
      for (unsigned position = 0; position < 4; ++position) {
        auto *gep =
            cast<GetElementPtrInst>(columnLoads[position]->getArgOperand(0));
        auto index = evaluate(gep->getOperand(1), lane);
        if (!require(index && index->constant >= 0 && index->constant < 128,
                     "column scale tile index"))
          return false;
        unsigned permuted = permute(index->constant, 1);
        if (position == 0)
          first = permuted;
        if (!require(permuted == first + position && first % 4 == 0,
                     "column scale vector contiguity"))
          return false;
      }
    }
    for (auto &owner : owners) {
      auto original = owner.second.constant / 2;
      owner.second.constant =
          2 * (int64_t(permute(original / columns, 0)) * columns +
               permute(original % columns, 1));
    }
    storeOrder = conversions;
    std::stable_sort(storeOrder.begin(), storeOrder.end(),
                     [&](CallInst *left, CallInst *right) {
                       return owners.at({0, left}).constant <
                              owners.at({0, right}).constant;
                     });
    CallInst *first = storeOrder.front();
    for (unsigned bit = 0; bit < 8; ++bit)
      laneCoefficients[bit] = owners.at({1u << bit, first}).constant -
                              owners.at({0, first}).constant;
    for (unsigned lane = 0; lane < 256; ++lane) {
      int64_t laneOffset = 0;
      for (unsigned bit = 0; bit < 8; ++bit)
        if (lane & (1u << bit))
          laneOffset += laneCoefficients[bit];
      for (unsigned position = 0; position < 64; ++position) {
        auto *conversion = storeOrder[position];
        auto address = owners.at({lane, conversion}).constant;
        int64_t rowOffset = owners.at({0, conversion}).constant / (2 * columns);
        for (unsigned bit = 0; bit < 8; ++bit)
          if (lane & (1u << bit))
            rowOffset += laneCoefficients[bit] / (2 * columns);
        if (!require(address / (2 * columns) == rowOffset,
                     "direct store row-mask coordinate"))
          return false;
        if (!require(
                address == owners.at({0, conversion}).constant + laneOffset &&
                    address == owners.at({lane, storeOrder[position / 4 * 4]})
                                       .constant +
                                   2 * (position % 4) &&
                    (position % 4 != 0 || address % 8 == 0),
                "direct vector-store ownership"))
          return false;
      }
    }
    for (CallInst *conversion : storeOrder)
      offsets[conversion] = owners.at({0, conversion}).constant;
    for (Instruction *instruction = conversions.back()->getNextNode();
         instruction; instruction = instruction->getNextNode()) {
      if (auto *fence = dyn_cast<FenceInst>(instruction)) {
        if (!require(
                fence->getSyncScopeID() ==
                        pattern.function.getContext().getOrInsertSyncScopeID(
                            "block") &&
                    (fence->getOrdering() == AtomicOrdering::Acquire ||
                     fence->getOrdering() == AtomicOrdering::Release),
                "output fence scope"))
          return false;
      } else if (instruction->mayHaveSideEffects() &&
                 !isa<ReturnInst>(instruction) &&
                 !llvm::is_contained(sharedStores, instruction) &&
                 !llvm::is_contained(globalStores, instruction) &&
                 called(instruction) != "llvm.mxc.barrier") {
        return require(false, "unknown output side effect");
      }
      for (User *user : instruction->users())
        if (!require(isa<Instruction>(user) &&
                         cast<Instruction>(user)->getParent() == pattern.exit,
                     "escaping output temporary"))
          return false;
    }
    return true;
  }

  Value *emitPermutation(IRBuilder<> &builder, Value *index, unsigned operand) {
    auto *type = cast<IntegerType>(index->getType());
    auto constant = [&](int64_t value) {
      return ConstantInt::get(type, value);
    };
    Value *base = builder.CreateAnd(index, constant(-128));
    if (operand == 0) {
      Value *low = builder.CreateAnd(index, constant(3));
      Value *middle = builder.CreateLShr(builder.CreateAnd(index, constant(96)),
                                         constant(3));
      Value *high = builder.CreateShl(builder.CreateAnd(index, constant(28)),
                                      constant(2));
      return builder.CreateOr(
          base, builder.CreateOr(low, builder.CreateOr(middle, high)),
          "mm.row.permuted");
    }
    Value *low =
        builder.CreateShl(builder.CreateAnd(index, constant(31)), constant(2));
    Value *high =
        builder.CreateLShr(builder.CreateAnd(index, constant(96)), constant(5));
    return builder.CreateOr(base, builder.CreateOr(low, high),
                            "mm.column.permuted");
  }

  void emitCtaOrder() {
    if (groupRows == 1)
      return;
    SmallVector<Use *> originalUses;
    for (Use &use : block->uses())
      originalUses.push_back(&use);
    IRBuilder<> builder(block->getNextNode());
    unsigned rowTiles = ((rows + 127) / 128), columnTiles = columns / 128;
    Value *group =
        builder.CreateUDiv(block, builder.getInt32(groupRows * columnTiles));
    Value *first = builder.CreateMul(group, builder.getInt32(groupRows));
    Value *count = builder.getInt32(groupRows);
    if (rowTiles % groupRows) {
      auto minimum = pattern.function.getParent()->getOrInsertFunction(
          "llvm.umin.i32", builder.getInt32Ty(), builder.getInt32Ty(),
          builder.getInt32Ty());
      count = builder.CreateCall(
          minimum,
          {builder.CreateSub(builder.getInt32(rowTiles), first), count});
    }
    Value *within =
        builder.CreateURem(block, builder.getInt32(groupRows * columnTiles));
    Value *row = builder.CreateAdd(first, builder.CreateURem(within, count));
    Value *column = builder.CreateUDiv(within, count);
    Value *mapped = builder.CreateAdd(
        row, builder.CreateMul(column, builder.getInt32(rowTiles)),
        "mm.cta.mapped");
    for (Use *use : originalUses)
      use->set(mapped);
    errs() << "grouped_ctas=" << rowTiles * columnTiles
           << " group_rows=" << groupRows << "\n";
  }

  void rematerializeTailIndices(BasicBlock *tail) {
    SmallVector<Instruction *> consumers;
    for (BasicBlock *block : {tail, pattern.exit})
      for (Instruction &instruction : *block)
        consumers.push_back(&instruction);

    IRBuilder<> builder(&*tail->getFirstInsertionPt());
    auto *type = FunctionType::get(builder.getInt32Ty(),
                                   {builder.getInt32Ty()}, false);
    // A tied empty asm preserves the lane value while preventing CSE from
    // keeping its derived output indices live across the steady-state loop.
    auto *identity = InlineAsm::get(type, "", "=r,0", true);
    auto *thread = builder.CreateCall(identity, {pattern.thread},
                                      "mm.tail.thread");
    thread->setConvergent();
    thread->setDoesNotThrow();
    std::map<Value *, Value *> rematerialized = {{pattern.thread, thread}};
    std::function<Value *(Value *)> rebuild = [&](Value *value) -> Value * {
      auto found = rematerialized.find(value);
      if (found != rematerialized.end())
        return found->second;
      auto *instruction = dyn_cast<Instruction>(value);
      if (!instruction || instruction->getParent() != pattern.entry)
        return value;
      bool pure = isa<BinaryOperator>(instruction) ||
                  isa<CastInst>(instruction) ||
                  isa<GetElementPtrInst>(instruction) ||
                  isa<CmpInst>(instruction) || isa<SelectInst>(instruction) ||
                  called(instruction) == "llvm.umin.i32";
      if (!pure)
        return value;
      SmallVector<Value *> operands;
      bool changed = false;
      for (Value *operand : instruction->operands()) {
        Value *replacement = rebuild(operand);
        operands.push_back(replacement);
        changed |= replacement != operand;
      }
      if (!changed)
        return rematerialized[value] = value;
      Instruction *clone = instruction->clone();
      for (unsigned index = 0; index < operands.size(); ++index)
        clone->setOperand(index, operands[index]);
      builder.Insert(clone, "mm.tail.index");
      return rematerialized[value] = clone;
    };
    for (Instruction *consumer : consumers)
      for (Use &operand : consumer->operands())
        operand.set(rebuild(operand.get()));
  }

  void prefetchColumnScale(BasicBlock *tail) {
    unsigned mmaCount = 0;
    Instruction *insertion = nullptr;
    for (Instruction &instruction : *tail) {
      if (called(&instruction) == "llvm.mxc.mma.i32.16x16x16i8" &&
          ++mmaCount == 160) {
        insertion = instruction.getNextNode();
        break;
      }
    }
    assert(insertion && "validated tail has 256 MMA instructions");
    std::function<void(Value *)> moveDependencies = [&](Value *value) {
      auto *instruction = dyn_cast<Instruction>(value);
      if (!instruction || instruction->getParent() != pattern.exit)
        return;
      for (Value *operand : instruction->operands())
        moveDependencies(operand);
      instruction->moveBefore(insertion);
    };
    // The checked counted loop always reaches its tail, and contains no global
    // stores.
    moveDependencies(packedColumnScales);
  }

  void emit() {
    IRBuilder<> maskBuilder(pattern.entry->getTerminator());
    if (pattern.splits > 1) {
      auto id = pattern.function.getParent()->getOrInsertFunction(
          "llvm.mxc.block.id.y", maskBuilder.getInt32Ty());
      splitId = maskBuilder.CreateCall(id, {}, "mm.split.id");
    }
    auto predicateFunction = pattern.function.getParent()->getOrInsertFunction(
        "llvm.mxc.icmp.i64.i32", maskBuilder.getInt64Ty(),
        maskBuilder.getInt32Ty(), maskBuilder.getInt32Ty(),
        maskBuilder.getInt32Ty());
    Value *unmasked = maskBuilder.CreateCall(
        predicateFunction, {maskBuilder.getInt32(1), maskBuilder.getInt32(0),
                            maskBuilder.getInt32(34)});
    for (Packet packet : pattern.packets)
      packet.load->setArgOperand(2, unmasked);
    for (CallInst *load : rowLoads)
      load->setArgOperand(2, unmasked);
    for (CallInst *load : columnLoads)
      load->setArgOperand(2, unmasked);
    auto minimum = pattern.function.getParent()->getOrInsertFunction(
        "llvm.umin.i32", maskBuilder.getInt32Ty(), maskBuilder.getInt32Ty(),
        maskBuilder.getInt32Ty());
    for (unsigned operand = 0; operand < 2; ++operand) {
      for (GetElementPtrInst *gep : matrixRows[operand]) {
        IRBuilder<> builder(gep);
        auto *multiply = cast<BinaryOperator>(gep->getOperand(1));
        Value *row = emitPermutation(builder, multiply->getOperand(0), operand);
        if (operand == 0 && rows % 128)
          row = builder.CreateCall(minimum, {row, builder.getInt32(rows - 1)},
                                   "mm.clamped.input.row");
        gep->setOperand(1, builder.CreateMul(row, multiply->getOperand(1)));
      }
      for (GetElementPtrInst *gep : scaleRows[operand]) {
        IRBuilder<> builder(gep);
        Value *index = emitPermutation(builder, gep->getOperand(1), operand);
        if (operand == 0 && rows % 128)
          index = builder.CreateCall(
              minimum, {index, builder.getInt32(rows - rowScaleWidth)},
              "mm.clamped.scale.row");
        gep->setOperand(1, index);
      }
    }
    IRBuilder<> scaleBuilder(columnLoads.front());
    auto *floatVector = FixedVectorType::get(scaleBuilder.getFloatTy(), 4);
    auto vectorLoad = pattern.function.getParent()->getOrInsertFunction(
        "llvm.mxc.ldg.predicator.v4f32", floatVector,
        PointerType::get(scaleBuilder.getContext(), 1),
        scaleBuilder.getInt32Ty(), scaleBuilder.getInt64Ty(),
        scaleBuilder.getInt1Ty(), scaleBuilder.getInt1Ty(),
        scaleBuilder.getInt1Ty(), scaleBuilder.getInt1Ty());
    SmallVector<Value *> scaleArguments;
    for (Value *argument : columnLoads.front()->args())
      scaleArguments.push_back(argument);
    packedColumnScales =
        scaleBuilder.CreateCall(vectorLoad, scaleArguments, "mm.column.scales");
    for (unsigned position = 0; position < 4; ++position) {
      scaleBuilder.SetInsertPoint(columnLoads[position]);
      Value *scalar = scaleBuilder.CreateExtractElement(
          packedColumnScales, scaleBuilder.getInt32(position));
      columnLoads[position]->replaceAllUsesWith(scalar);
      columnLoads[position]->eraseFromParent();
    }
    SmallVector<Instruction *> obsolete;
    for (Instruction *instruction = conversions.back()->getNextNode();
         instruction; instruction = instruction->getNextNode())
      obsolete.push_back(instruction);
    for (Instruction *instruction : obsolete)
      instruction->dropAllReferences();
    for (Instruction *instruction : obsolete)
      instruction->eraseFromParent();
    IRBuilder<> builder(pattern.exit);
    bool elementAddress =
        pattern.splits > 1 ||
        uint64_t((rows + 127) / 128) * 128 * columns * 2 >= (1ULL << 31);
    unsigned addressUnit = elementAddress ? 1 : 2;
    auto *addressType = builder.getInt32Ty();
    auto addressConstant = [&](int64_t value) {
      return ConstantInt::get(addressType, value);
    };
    Value *row = builder.CreateSExtOrTrunc(
        builder.CreateSRem(block, builder.getInt32(((rows + 127) / 128))),
        addressType);
    Value *column = builder.CreateSExtOrTrunc(
        builder.CreateSDiv(block, builder.getInt32(((rows + 127) / 128))),
        addressType);
    Value *base = builder.CreateAdd(
        builder.CreateMul(
            row, addressConstant(uint64_t(128) * addressUnit * columns)),
        builder.CreateMul(column, addressConstant(128 * addressUnit)));
    Value *laneOffset = addressConstant(0);
    for (unsigned bit = 0; bit < 8; ++bit) {
      Value *laneBit = builder.CreateAnd(
          builder.CreateLShr(pattern.thread, builder.getInt32(bit)),
          builder.getInt32(1));
      laneOffset = builder.CreateAdd(
          laneOffset,
          builder.CreateMul(builder.CreateZExtOrTrunc(laneBit, addressType),
                            addressConstant(laneCoefficients[bit] /
                                            (elementAddress ? 2 : 1))));
    }
    base = builder.CreateAdd(base, laneOffset, "mm.output.base");
    Value *outputRowBase = builder.CreateMul(
        builder.CreateSRem(block, builder.getInt32((rows + 127) / 128)),
        builder.getInt32(128));
    for (unsigned bit = 0; bit < 8; ++bit) {
      Value *laneBit = builder.CreateAnd(
          builder.CreateLShr(pattern.thread, builder.getInt32(bit)),
          builder.getInt32(1));
      outputRowBase = builder.CreateAdd(
          outputRowBase,
          builder.CreateMul(laneBit, builder.getInt32(laneCoefficients[bit] /
                                                      (2 * columns))));
    }
    auto *vectorType = FixedVectorType::get(
        pattern.splits > 1 ? builder.getInt32Ty() : builder.getInt16Ty(), 4);
    Type *packedType =
        pattern.splits > 1
            ? static_cast<Type *>(FixedVectorType::get(builder.getInt16Ty(), 8))
            : builder.getInt64Ty();
    auto store = pattern.function.getParent()->getOrInsertFunction(
        pattern.splits > 1 ? "llvm.mxc.stg.predicator.v8i16"
                           : "llvm.mxc.stg.predicator.i64",
        builder.getVoidTy(), PointerType::get(builder.getContext(), 1),
        builder.getInt32Ty(), packedType, builder.getInt64Ty(),
        builder.getInt1Ty(), builder.getInt1Ty(), builder.getInt1Ty());
    auto predicate = pattern.function.getParent()->getOrInsertFunction(
        "llvm.mxc.icmp.i64.i32", builder.getInt64Ty(), builder.getInt32Ty(),
        builder.getInt32Ty(), builder.getInt32Ty());
    Value *mask =
        builder.CreateCall(predicate, {builder.getInt32(1), builder.getInt32(0),
                                       builder.getInt32(34)});
    for (unsigned position = 0; position < 64; position += 4) {
      Value *vector = PoisonValue::get(vectorType);
      for (unsigned element = 0; element < 4; ++element)
        vector = builder.CreateInsertElement(
            vector,
            pattern.splits > 1
                ? integerOutputs.at(storeOrder[position + element])
                : storeOrder[position + element],
            builder.getInt32(element));
      Value *offset = builder.CreateAdd(
          base, addressConstant(offsets.at(storeOrder[position]) /
                                (elementAddress ? 2 : 1)));
      Value *pointer;
      if (pattern.splits > 1) {
        Value *splitBase =
            builder.CreateMul(builder.CreateZExt(splitId, builder.getInt64Ty()),
                              builder.getInt64(uint64_t(rows) * columns));
        Value *index = builder.CreateAdd(
            splitBase, builder.CreateZExt(offset, builder.getInt64Ty()));
        pointer =
            builder.CreateGEP(builder.getInt32Ty(), pattern.function.getArg(4),
                              index, "mm.partial.pointer");
      } else if (elementAddress) {
        // Keep the nonnegative element index explicit across the vendor address
        // combiner.
        pointer =
            builder.CreateGEP(builder.getInt16Ty(), pattern.function.getArg(4),
                              builder.CreateZExt(offset, builder.getInt64Ty()),
                              "mm.output.pointer");
      } else {
        pointer =
            builder.CreateGEP(builder.getInt8Ty(), pattern.function.getArg(4),
                              offset, "mm.output.pointer");
      }
      Value *storeMask = mask;
      if (rows % 128) {
        Value *outputRow = builder.CreateAdd(
            outputRowBase,
            builder.getInt32(offsets.at(storeOrder[position]) / (2 * columns)));
        Value *active = builder.CreateZExt(
            builder.CreateICmpULT(outputRow, builder.getInt32(rows)),
            builder.getInt32Ty());
        storeMask = builder.CreateCall(
            predicate, {active, builder.getInt32(0), builder.getInt32(34)});
      }
      builder.CreateCall(store, {pointer, builder.getInt32(0),
                                 builder.CreateBitCast(vector, packedType),
                                 storeMask, builder.getTrue(),
                                 builder.getFalse(), builder.getFalse()});
    }
    builder.CreateRetVoid();
    if (pattern.splits > 1) {
      std::set<Instruction *> live;
      std::function<void(Value *)> visit = [&](Value *value) {
        auto *instruction = dyn_cast<Instruction>(value);
        if (!instruction || instruction->getParent() != pattern.exit ||
            !live.insert(instruction).second)
          return;
        for (Value *operand : instruction->operands())
          visit(operand);
      };
      for (Instruction &instruction : *pattern.exit)
        if (called(&instruction) == "llvm.mxc.stg.predicator.v8i16" ||
            isa<ReturnInst>(instruction))
          visit(&instruction);
      SmallVector<Instruction *> dead;
      for (Instruction &instruction : *pattern.exit)
        if (!live.count(&instruction))
          dead.push_back(&instruction);
      for (Instruction *instruction : dead)
        instruction->dropAllReferences();
      for (Instruction *instruction : dead)
        instruction->eraseFromParent();
    }
  }
};
