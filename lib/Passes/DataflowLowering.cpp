#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRDATAFLOWLOWERINGPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;
namespace arith = mlir::arith;
namespace memref = mlir::memref;
namespace scf = mlir::scf;

//===----------------------------------------------------------------------===//
// TypeConverter: Field/Temp -> memref (shape derived from bounds)
//===----------------------------------------------------------------------===//

static MemRefType memrefTypeFromBounds(Type elemTy, BoundsAttr bounds) {
  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  SmallVector<int64_t> shape;
  shape.reserve(lb.size());
  for (auto [l, u] : llvm::zip(lb, ub))
    shape.push_back(u - l);
  return MemRefType::get(shape, elemTy);
}

struct NeptuneIRTypeConverter final : TypeConverter {
  explicit NeptuneIRTypeConverter(MLIRContext *ctx) {
    addConversion([](Type t) { return t; });

    addConversion([&](FieldType ty) -> std::optional<Type> {
      auto lb = ty.getBounds().getLb().asArrayRef();
      auto ub = ty.getBounds().getUb().asArrayRef();
      SmallVector<int64_t> shape;
      for (auto [l, u] : llvm::zip(lb, ub))
        shape.push_back(u - l);
      return MemRefType::get(shape, ty.getElementType());
    });

    addConversion([&](TempType ty) -> std::optional<Type> {
      auto lb = ty.getBounds().getLb().asArrayRef();
      auto ub = ty.getBounds().getUb().asArrayRef();
      SmallVector<int64_t> shape;
      for (auto [l, u] : llvm::zip(lb, ub))
        shape.push_back(u - l);
      return MemRefType::get(shape, ty.getElementType());
    });

    addTargetMaterialization([&](OpBuilder &b, Type dstType, ValueRange inputs,
                                 Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      auto dst = dyn_cast<MemRefType>(dstType);
      auto src = dyn_cast<MemRefType>(inputs[0].getType());
      if (!dst || !src)
        return {};
      if (dst == src)
        return inputs[0];
      if (memref::CastOp::areCastCompatible(src, dst))
        return b.create<memref::CastOp>(loc, dst, inputs[0]).getResult();
      return {};
    });

    addSourceMaterialization([&](OpBuilder &b, Type dstType, ValueRange inputs,
                                 Location loc) -> Value {
      // 通常和 target materialization 一样写即可
      if (inputs.size() != 1)
        return {};
      auto dst = dyn_cast<MemRefType>(dstType);
      auto src = dyn_cast<MemRefType>(inputs[0].getType());
      if (!dst || !src)
        return {};
      if (dst == src)
        return inputs[0];
      if (memref::CastOp::areCastCompatible(src, dst))
        return b.create<memref::CastOp>(loc, dst, inputs[0]).getResult();
      return {};
    });
  }
};

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

static Value cIndex(Location loc, int64_t v, OpBuilder &b) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

// idx = (iv - baseLb) + offset
static Value makeLocalIndex(Location loc, OpBuilder &b, Value iv,
                            int64_t baseLb, int64_t offset) {
  Value idx = iv;
  if (baseLb != 0) {
    idx = b.create<arith::SubIOp>(loc, idx, cIndex(loc, baseLb, b));
  }
  if (offset != 0) {
    idx = b.create<arith::AddIOp>(loc, idx, cIndex(loc, offset, b));
  }
  return idx;
}

//===----------------------------------------------------------------------===//
// 1) wrap/unwrap/load : no-op after type conversion
//===----------------------------------------------------------------------===//

struct WrapLowering : OpConversionPattern<WrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(WrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getBuffer());
    return success();
  }
};

struct UnwrapLowering : OpConversionPattern<UnwrapOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnwrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

struct LoadLowering : OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 2) store : memref.copy (optionally subview src/dst with bounds)
//===----------------------------------------------------------------------===//

struct StoreLowering : OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getValue();    // converted memref
    Value dst = adaptor.getVarField(); // converted memref

    // Full copy
    if (!op.getBounds()) {
      rewriter.replaceOpWithNewOp<memref::CopyOp>(op, src, dst);
      return success();
    }

    // Subdomain copy: bounds are in LOGICAL coordinates.
    // Need to convert to each memref's 0-based coordinates by subtracting base
    // lb.
    auto b = *op.getBounds();

    auto srcTyN = dyn_cast<TempType>(op.getValue().getType());
    auto dstTyN = dyn_cast<FieldType>(op.getVarField().getType());
    if (!srcTyN || !dstTyN)
      return rewriter.notifyMatchFailure(
          op, "store expects TempType -> FieldType before conversion");

    auto srcLb = srcTyN.getBounds().getLb().asArrayRef();
    auto dstLb = dstTyN.getBounds().getLb().asArrayRef();

    auto lb = b.getLb().asArrayRef();
    auto ub = b.getUb().asArrayRef();
    if (lb.size() != ub.size() || lb.size() != srcLb.size() ||
        lb.size() != dstLb.size())
      return rewriter.notifyMatchFailure(op, "bounds rank mismatch");

    SmallVector<OpFoldResult> srcOffsets, dstOffsets, sizes, strides;
    for (size_t i = 0; i < lb.size(); ++i) {
      int64_t offSrc = lb[i] - srcLb[i];
      int64_t offDst = lb[i] - dstLb[i];
      int64_t sz = ub[i] - lb[i];
      srcOffsets.push_back(rewriter.getIndexAttr(offSrc));
      dstOffsets.push_back(rewriter.getIndexAttr(offDst));
      sizes.push_back(rewriter.getIndexAttr(sz));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    auto srcView = rewriter.create<memref::SubViewOp>(loc, src, srcOffsets,
                                                      sizes, strides);
    auto dstView = rewriter.create<memref::SubViewOp>(loc, dst, dstOffsets,
                                                      sizes, strides);

    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, srcView, dstView);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 3) Drop trivial unrealized cast when it becomes memref->same memref
//===----------------------------------------------------------------------===//

struct ForwardUnrealizedCastToMemref
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getResults().size() != 1)
      return failure();

    auto *tc = getTypeConverter();
    if (!tc)
      return failure();

    Type convertedResTy = tc->convertType(op.getResult(0).getType());
    if (!convertedResTy)
      return failure();

    Value in = adaptor.getInputs().front();
    if (in.getType() != convertedResTy)
      return failure();

    rewriter.replaceOp(op, in);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 4) apply : loop nest + clone scalar body; access lowered inline; yield ->
// store
//===----------------------------------------------------------------------===//

struct ApplyToSCFForLowering : OpConversionPattern<ApplyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto bounds = op.getBounds();
    ArrayRef<int64_t> lbs = bounds.getLb().asArrayRef();
    ArrayRef<int64_t> ubs = bounds.getUb().asArrayRef();
    unsigned rank = lbs.size();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "0-D apply not supported");

    // converted result type
    auto *tc = getTypeConverter();
    auto resTy =
        dyn_cast<MemRefType>(tc->convertType(op.getResult().getType()));
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result not converted to memref");

    // alloc result
    rewriter.setInsertionPoint(op);
    auto out = rewriter.create<memref::AllocOp>(loc, resTy);
    // 默认 copy-through：out = input0
    ValueRange inputs = adaptor.getOperands();
    Value in0 = inputs[0];
    if (in0.getType() != out.getType())
      in0 = rewriter.create<memref::CastOp>(loc, out.getType(), in0);
    rewriter.create<memref::CopyOp>(loc, in0, out);
    // build loop nest in logical index space [lb, ub)
    SmallVector<scf::ForOp> loops;
    loops.reserve(rank);

    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // outer loop
    Value lb0 = rewriter.create<arith::ConstantIndexOp>(loc, lbs[0]);
    Value ub0 = rewriter.create<arith::ConstantIndexOp>(loc, ubs[0]);
    auto outer = rewriter.create<scf::ForOp>(loc, lb0, ub0, c1);
    loops.push_back(outer);

    scf::ForOp cur = outer;
    for (unsigned d = 1; d < rank; ++d) {
      rewriter.setInsertionPointToStart(cur.getBody());
      Value lbd = rewriter.create<arith::ConstantIndexOp>(loc, lbs[d]);
      Value ubd = rewriter.create<arith::ConstantIndexOp>(loc, ubs[d]);
      auto inner = rewriter.create<scf::ForOp>(loc, lbd, ubd, c1);
      loops.push_back(inner);
      cur = inner;
    }

    scf::ForOp innermost = loops.back();
    Operation *term = innermost.getBody()->getTerminator();
    rewriter.setInsertionPoint(term);

    // map region args: ^bb0(%i0, %i1, ...) to ivs
    Block &body = op.getBody().front();
    IRMapping map;
    unsigned numInputs = op.getInputs().size();
    unsigned firstInputArg = body.getNumArguments() - numInputs;
    unsigned numIndexArgs = firstInputArg;

    // 1) index args -> loop IVs
    if (loops.size() != numIndexArgs)
      return rewriter.notifyMatchFailure(
          op, "apply region index-arg count != loop nest depth");

    for (unsigned d = 0; d < numIndexArgs; ++d)
      map.map(body.getArgument(d), loops[d].getInductionVar());

    // 2) input temp args -> converted memrefs
    for (unsigned i = 0; i < numInputs; ++i)
      map.map(body.getArgument(firstInputArg + i), adaptor.getInputs()[i]);

    auto findInputIndex = [&](Value inMem) -> std::optional<unsigned> {
      for (unsigned i = 0; i < numInputs; ++i)
        if (inMem == adaptor.getInputs()[i])
          return i;
      return std::nullopt;
    };

    auto lowerAccessOp = [&](AccessOp acc) -> LogicalResult {
      // 关键：在 acc 位置插入，保证 dominance
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(acc);

      Location aloc = acc.getLoc();
      // 关键：别用 acc.getInput() / typed accessor
      Value inMem = acc->getOperand(0);
      // 你的 findInputIndex 仍然可用（建议用 strip cast 更鲁棒，先不加）
      auto idxOpt = findInputIndex(inMem);
      if (!idxOpt)
        return failure();
      unsigned i = *idxOpt;

      // 通过 op.getInputs()[i] 拿原始 TempType 的 lb（不要从 inMem 类型推）
      auto inTempTy = cast<TempType>(op.getInputs()[i].getType());
      ArrayRef<int64_t> inLb = inTempTy.getBounds().getLb().asArrayRef();
      ArrayRef<int64_t> offs = acc.getOffsetsAttr().asArrayRef();

      SmallVector<Value> idx;
      idx.reserve(rank);
      for (unsigned d = 0; d < rank; ++d) {
        Value iv = loops[d].getInductionVar();

        if (offs[d] != 0)
          iv = rewriter.create<arith::AddIOp>(aloc, iv,
                                              cIndex(aloc, offs[d], rewriter));
        if (inLb[d] != 0)
          iv = rewriter.create<arith::SubIOp>(aloc, iv,
                                              cIndex(aloc, inLb[d], rewriter));

        idx.push_back(iv);
      }

      Value v = rewriter.create<memref::LoadOp>(aloc, inMem, idx);
      rewriter.replaceOp(acc, v);
      return success();
    };

    // clone body (handle access specially)
    for (Operation &nested : body.without_terminator()) {
      if (auto acc = dyn_cast<AccessOp>(nested)) {
        auto inTemp = cast<TempType>(acc.getInput().getType());
        ArrayRef<int64_t> inLb = inTemp.getBounds().getLb().asArrayRef();
        ArrayRef<int64_t> offs = acc.getOffsetsAttr().asArrayRef();

        Value inMem = map.lookup(acc.getInput());

        SmallVector<Value> idx;
        idx.reserve(rank);
        for (unsigned d = 0; d < rank; ++d) {
          Value iv = loops[d].getInductionVar();
          Value off =
              (offs[d] == 0)
                  ? Value{}
                  : rewriter.create<arith::ConstantIndexOp>(loc, offs[d]);
          Value logical = (offs[d] == 0)
                              ? iv
                              : rewriter.create<arith::AddIOp>(loc, iv, off);

          // IMPORTANT: physical = logical - lb
          if (inLb[d] != 0) {
            Value clb = rewriter.create<arith::ConstantIndexOp>(loc, inLb[d]);
            logical = rewriter.create<arith::SubIOp>(loc, logical, clb);
          }
          idx.push_back(logical);
        }

        Value v = rewriter.create<memref::LoadOp>(loc, inMem, idx);
        map.map(acc.getResult(), v);
        continue;
      }

      Operation *cloned = rewriter.clone(nested, map);
      SmallVector<AccessOp> accs;
      cloned->walk([&](AccessOp a) { accs.push_back(a); });

      for (AccessOp a : accs) {
        if (failed(lowerAccessOp(a)))
          return rewriter.notifyMatchFailure(
              op, "failed to lower nested access ops");
      }
      for (auto [o, n] : llvm::zip(nested.getResults(), cloned->getResults()))
        map.map(o, n);
    }

    // yield -> store scalar to out at (iv - lb)
    auto y = cast<YieldOp>(body.getTerminator());
    if (y.getNumOperands() != 1)
      return y.emitOpError("MVP: only single-scalar yield supported");

    Value scalar = map.lookup(y.getOperand(0));
    auto outTemp = cast<TempType>(op.getResult().getType());
    auto outLb = outTemp.getBounds().getLb().asArrayRef();
    SmallVector<Value> outIdx;
    for (unsigned d = 0; d < rank; ++d) {
      Value iv = loops[d].getInductionVar();
      if (outLb[d] != 0) {
        Value clb = rewriter.create<arith::ConstantIndexOp>(loc, outLb[d]);
        iv = rewriter.create<arith::SubIOp>(loc, iv, clb);
      }
      outIdx.push_back(iv);
    }

    rewriter.create<memref::StoreOp>(loc, scalar, out, outIdx);
    rewriter.replaceOp(op, out.getResult());
    return success();
  }
};

struct ApplyToGpuLaunchLowering : OpConversionPattern<ApplyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto bounds = op.getBounds();
    ArrayRef<int64_t> lbs = bounds.getLb().asArrayRef();
    ArrayRef<int64_t> ubs = bounds.getUb().asArrayRef();
    unsigned rank = lbs.size();
    if (rank != 1)
      return rewriter.notifyMatchFailure(
          op, "MVP gpu backend: only rank-1 apply supported");

    auto *tc = getTypeConverter();
    auto resTy =
        dyn_cast<MemRefType>(tc->convertType(op.getResult().getType()));
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result not converted to memref");

    // ---- 输出 buffer：这里先用 memref.alloc（你后续建议换 gpu.alloc
    // 或统一内存 allocator）
    rewriter.setInsertionPoint(op);
    Value out = rewriter.create<memref::AllocOp>(loc, resTy);

    // launch config: block=256 threads, grid = ceildiv(extent, 256)
    int64_t extent = ubs[0] - lbs[0];
    Value cExtent = rewriter.create<arith::ConstantIndexOp>(loc, extent);
    Value cBlock = rewriter.create<arith::ConstantIndexOp>(loc, 256);

    // grid = (extent + block - 1) / block
    Value cBlockMinus1 = rewriter.create<arith::ConstantIndexOp>(loc, 255);
    Value num = rewriter.create<arith::AddIOp>(loc, cExtent, cBlockMinus1);
    Value grid = rewriter.create<arith::DivUIOp>(loc, num, cBlock);

    // gpu.launch (grid x block)
    auto launch = rewriter.create<gpu::LaunchOp>(
        loc,
        /*gridSizeX=*/grid,
        /*gridSizeY=*/rewriter.create<arith::ConstantIndexOp>(loc, 1),
        /*gridSizeZ=*/rewriter.create<arith::ConstantIndexOp>(loc, 1),
        /*blockSizeX=*/cBlock,
        /*blockSizeY=*/rewriter.create<arith::ConstantIndexOp>(loc, 1),
        /*blockSizeZ=*/rewriter.create<arith::ConstantIndexOp>(loc, 1));

    // Build kernel body
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&launch.getBody().front());

      Value bx = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
      Value tx = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);

      // linear = bx*block + tx
      Value mul = rewriter.create<arith::MulIOp>(loc, bx, cBlock);
      Value tid = rewriter.create<arith::AddIOp>(loc, mul, tx);

      // if (tid < extent)
      Value pred = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, tid, cExtent);
      auto ifOp =
          rewriter.create<scf::IfOp>(loc, pred, /*withElseRegion=*/false);

      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

      // logical iv = lb + tid
      Value cLb = rewriter.create<arith::ConstantIndexOp>(loc, lbs[0]);
      Value iv = rewriter.create<arith::AddIOp>(loc, cLb, tid);

      // clone apply region scalar body：暂时只支持 access + arith
      Block &body = op.getBody().front();
      IRMapping map;
      unsigned numInputs = op.getInputs().size();
      unsigned firstInputArg = body.getNumArguments() - numInputs;
      unsigned numIndexArgs = firstInputArg;

      // MVP gpu：你目前只支持 1D，所以这里要求 numIndexArgs==1
      if (numIndexArgs != 1)
        return rewriter.notifyMatchFailure(
            op, "MVP gpu: apply region must have exactly 1 index arg");

      map.map(body.getArgument(0), iv);

      // inputs
      for (unsigned i = 0; i < numInputs; ++i)
        map.map(body.getArgument(firstInputArg + i), adaptor.getInputs()[i]);

      for (Operation &nested : body.without_terminator()) {
        if (auto acc = dyn_cast<AccessOp>(nested)) {
          auto inTemp = cast<TempType>(acc.getInput().getType());
          int64_t inLb0 = inTemp.getBounds().getLb().asArrayRef()[0];
          int64_t off0 = acc.getOffsetsAttr().asArrayRef()[0];

          Value inMem = map.lookup(acc.getInput());
          Value logical = iv;
          if (off0 != 0) {
            Value cOff = rewriter.create<arith::ConstantIndexOp>(loc, off0);
            logical = rewriter.create<arith::AddIOp>(loc, logical, cOff);
          }
          // physical = logical - inLb
          if (inLb0 != 0) {
            Value cInLb = rewriter.create<arith::ConstantIndexOp>(loc, inLb0);
            logical = rewriter.create<arith::SubIOp>(loc, logical, cInLb);
          }

          Value v =
              rewriter.create<memref::LoadOp>(loc, inMem, ValueRange{logical});
          map.map(acc.getResult(), v);
          continue;
        }

        Operation *cloned = rewriter.clone(nested, map);
        for (auto [o, n] : llvm::zip(nested.getResults(), cloned->getResults()))
          map.map(o, n);
      }

      auto y = cast<YieldOp>(body.getTerminator());
      if (y.getNumOperands() != 1)
        return y.emitOpError("MVP gpu: only single-scalar yield supported");

      Value scalar = map.lookup(y.getOperand(0));

      // out index = tid (because output memref is [0, extent))
      rewriter.create<memref::StoreOp>(loc, scalar, out, ValueRange{tid});

      rewriter.create<scf::YieldOp>(loc);
    }

    rewriter.setInsertionPointAfter(launch);
    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 5) reduce : build nested loops over logical bounds, sum over memref
//===----------------------------------------------------------------------===//

static Value makeZero(Location loc, Type elemTy, OpBuilder &b) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(loc, elemTy,
                                       b.getFloatAttr(elemTy, 0.0));
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return b.create<arith::ConstantOp>(loc, elemTy,
                                       b.getIntegerAttr(elemTy, 0));
  return {};
}

static Value buildReduceLoops(Location loc, OpBuilder &b, Value mem,
                              Type elemTy, ArrayRef<int64_t> domainLb,
                              ArrayRef<int64_t> domainUb,
                              ArrayRef<int64_t> baseLb, unsigned dim,
                              SmallVector<Value> &logicalIvs, Value accInit) {
  Value lower = cIndex(loc, domainLb[dim], b);
  Value upper = cIndex(loc, domainUb[dim], b);
  Value step = cIndex(loc, 1, b);

  auto loop = b.create<scf::ForOp>(
      loc, lower, upper, step, ValueRange{accInit},
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs) {
        SmallVector<Value> newIvs(logicalIvs);
        newIvs.push_back(iv);

        Value acc = iterArgs.front();

        if (dim + 1 == domainLb.size()) {
          // load at (iv - baseLb)
          SmallVector<Value> idxs;
          idxs.reserve(newIvs.size());
          for (unsigned d = 0; d < newIvs.size(); ++d)
            idxs.push_back(makeLocalIndex(nloc, nb, newIvs[d], baseLb[d], 0));

          Value v = nb.create<memref::LoadOp>(nloc, mem, idxs);

          Value next;
          if (isa<FloatType>(elemTy))
            next = nb.create<arith::AddFOp>(nloc, acc, v);
          else
            next = nb.create<arith::AddIOp>(nloc, acc, v);

          nb.create<scf::YieldOp>(nloc, next);
          return;
        }

        Value inner = buildReduceLoops(nloc, nb, mem, elemTy, domainLb,
                                       domainUb, baseLb, dim + 1, newIvs, acc);
        nb.create<scf::YieldOp>(nloc, inner);
      });

  return loop.getResult(0);
}

struct ReduceLowering : OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // MVP: kind == "sum"
    if (op.getKind() != "sum")
      return rewriter.notifyMatchFailure(
          op, "MVP reduce only supports kind=\"sum\"");

    auto tempTy = dyn_cast<TempType>(op.getInput().getType());
    if (!tempTy)
      return rewriter.notifyMatchFailure(
          op, "reduce input must be TempType pre-conversion");

    Value mem = adaptor.getInput(); // converted memref
    auto memTy = dyn_cast<MemRefType>(mem.getType());
    if (!memTy)
      return rewriter.notifyMatchFailure(
          op, "reduce input did not convert to memref");

    Type elemTy = memTy.getElementType();
    Value zero = makeZero(loc, elemTy, rewriter);
    if (!zero)
      return rewriter.notifyMatchFailure(op, "unsupported reduce element type");

    auto baseLb = tempTy.getBounds().getLb().asArrayRef();
    auto baseUb = tempTy.getBounds().getUb().asArrayRef();

    // Domain to reduce over: op.bounds ? op.bounds : tempTy.bounds
    SmallVector<int64_t> domainLb, domainUb;
    if (auto b = op.getBoundsAttr()) {
      auto lb = b.getLb().asArrayRef();
      auto ub = b.getUb().asArrayRef();
      domainLb.assign(lb.begin(), lb.end());
      domainUb.assign(ub.begin(), ub.end());
    } else {
      domainLb.assign(baseLb.begin(), baseLb.end());
      domainUb.assign(baseUb.begin(), baseUb.end());
    }

    if (domainLb.size() != domainUb.size() || domainLb.size() != baseLb.size())
      return rewriter.notifyMatchFailure(op, "bounds rank mismatch in reduce");

    SmallVector<Value> logicalIvs;
    Value acc =
        buildReduceLoops(loc, rewriter, mem, elemTy, domainLb, domainUb, baseLb,
                         /*dim=*/0, logicalIvs, zero);

    rewriter.replaceOp(op, acc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 6) as_tensor / from_tensor (optional safety): keep as unrealized casts
//    让后续 bufferization/其他 pass 处理真正 tensorize/de-tensorize。
//===----------------------------------------------------------------------===//

struct AsTensorLowering : OpConversionPattern<AsTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AsTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resTy = op.getResult().getType(); // tensor type stays tensor
    auto cast = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), resTy,
                                                            adaptor.getInput());
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct FromTensorLowering : OpConversionPattern<FromTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResTy)
      return rewriter.notifyMatchFailure(
          op, "cannot convert from_tensor result type");
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), convertedResTy, adaptor.getInput());
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NeptuneIRDataflowLoweringPass final
    : public mlir::Neptune::NeptuneIR::impl::NeptuneIRDataflowLoweringPassBase<
          NeptuneIRDataflowLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();
    auto module = dyn_cast<ModuleOp>(root);
    if (!module) {
      root->emitError() << "neptune-ir-dataflow-lowering expects ModuleOp";
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = module.getContext();
    NeptuneIRTypeConverter typeConverter(ctx);

    ConversionTarget target(*ctx);

    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>(); // 仅用于允许 tensor
                                                     // type/ops 出现

    target.addIllegalDialect<NeptuneIRDialect>();

    // Make unknown ops legal iff their types are legal.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    // func.func signature/body type conversion
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
      return typeConverter.isSignatureLegal(f.getFunctionType()) &&
             typeConverter.isLegal(&f.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getOperandTypes()) &&
             typeConverter.isLegal(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp r) {
      return typeConverter.isLegal(r.getOperandTypes());
    });

    RewritePatternSet patterns(ctx);

    // Convert function signatures / calls / returns
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // NeptuneIR op lowerings
    patterns.add<WrapLowering, UnwrapLowering, LoadLowering, StoreLowering,
                 ForwardUnrealizedCastToMemref, ReduceLowering,
                 AsTensorLowering, FromTensorLowering>(typeConverter, ctx);

    if (backend == DataflowBackend::cpu) {
      patterns.add<ApplyToSCFForLowering>(typeConverter, ctx);
      // patterns.add<ReduceToLoopsLowering>(...)  // 你之前那套 ND reduction
      // 也应放这里
    } else if (backend == DataflowBackend::gpu) {
      patterns.add<ApplyToGpuLaunchLowering>(typeConverter, ctx);
      // Reduce GPU：MVP 可先 lower 到 runtime call（例如
      // _neptune_rt_reduce_sum）
    } else {
      module.emitError() << "unknown backend: " << backend.ValueStr
                         << " (expected cpu|gpu)";
      signalPassFailure();
      return;
    }

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
