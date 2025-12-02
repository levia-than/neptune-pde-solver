/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-11-29 12:08:15
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-29 15:31:40
 * @FilePath: /neptune-pde-solver/lib/Passes/NeptuneIRStencilToSCF.cpp
 * @Description:
 *
 * Copyright (c) 2025 by leviathan, All Rights Reserved.
 */
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"

#include "Passes/NeptuneIRPasses.h" // 由 TableGen 生成的头

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyFullConversion

using namespace mlir;

namespace {
using namespace mlir::Neptune::NeptuneIR;
/// TypeConverter：Field / Temp → memref
struct NeptuneIRTypeConverter : public TypeConverter {
  NeptuneIRTypeConverter(MLIRContext *ctx) {
    // 先来个「默认 identity」
    addConversion([](Type type) { return type; });

    // FieldType -> memref
    addConversion([](FieldType fieldTy) -> std::optional<Type> {
      auto elemTy = fieldTy.getElementType();
      auto bounds = fieldTy.getBounds();
      auto lb = bounds.getLb().asArrayRef();
      auto ub = bounds.getUb().asArrayRef();

      SmallVector<int64_t> shape;
      shape.reserve(lb.size());
      for (auto [l, u] : llvm::zip(lb, ub))
        shape.push_back(u - l);

      return MemRefType::get(shape, elemTy);
    });

    // TempType -> memref（同样用 bounds 推 shape）
    addConversion([](TempType tempTy) -> std::optional<Type> {
      auto elemTy = tempTy.getElementType();
      auto bounds = tempTy.getBounds();
      auto lb = bounds.getLb().asArrayRef();
      auto ub = bounds.getUb().asArrayRef();

      SmallVector<int64_t> shape;
      shape.reserve(lb.size());
      for (auto [l, u] : llvm::zip(lb, ub))
        shape.push_back(u - l);

      return MemRefType::get(shape, elemTy);
    });
  }
};

/// neptune_ir.wrap %buffer : memref<...> -> !neptune_ir.field<...>
/// 降成「直接转发」：
struct WrapOpLowering : public OpConversionPattern<WrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // buffer 已经经过 TypeConverter，是 memref 了
    rewriter.replaceOp(op, adaptor.getBuffer());
    return success();
  }
};

/// neptune_ir.unwrap %field : field -> memref
/// 同样是直接转发
struct UnwrapOpLowering : public OpConversionPattern<UnwrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnwrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

/// neptune_ir.load %field : field -> temp
/// FieldType / TempType 都映射到同一个 memref，所以这里也只是别名
struct LoadOpLowering : public OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

/// neptune_ir.store %temp to %field {bounds?}
/// 先忽略部分写，只做完整 memref.copy
struct StoreOpLowering : public OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // value / var_field 都已经被转换成 memref 了
    Location loc = op.getLoc();
    Value src = adaptor.getValue();
    Value dst = adaptor.getVarField();

    if (auto bounds = op.getBounds()) {
      ArrayRef<int64_t> lbs = bounds->getLb().asArrayRef();
      ArrayRef<int64_t> ubs = bounds->getUb().asArrayRef();
      if (lbs.size() != ubs.size())
        return rewriter.notifyMatchFailure(op, "lb/ub rank mismatch");

      SmallVector<OpFoldResult> offsets, sizes, strides;
      offsets.reserve(lbs.size());
      sizes.reserve(lbs.size());
      strides.reserve(lbs.size());
      for (size_t i = 0; i < lbs.size(); ++i) {
        offsets.push_back(rewriter.getIndexAttr(lbs[i]));
        sizes.push_back(rewriter.getIndexAttr(ubs[i] - lbs[i]));
        strides.push_back(rewriter.getIndexAttr(1));
      }
      auto subview =
          rewriter.create<memref::SubViewOp>(loc, dst, offsets, sizes,
                                             strides);
      rewriter.replaceOpWithNewOp<memref::CopyOp>(op, src, subview);
      return success();
    }

    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, src, dst);
    return success();
  }
};

/// 核心：neptune_ir.apply 降成 loop nest + memref.load/store
struct ApplyOpLowering : public OpConversionPattern<ApplyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto bounds = op.getBounds();
    DenseI64ArrayAttr lbAttr = bounds.getLb();
    DenseI64ArrayAttr ubAttr = bounds.getUb();
    ArrayRef<int64_t> lbs = lbAttr.asArrayRef();
    ArrayRef<int64_t> ubs = ubAttr.asArrayRef();

    unsigned rank = lbs.size();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "0-D apply is not supported yet");

    // 用 TypeConverter 看一下 result 被转换成什么类型
    auto *typeConverter = getTypeConverter();
    if (!typeConverter)
      return rewriter.notifyMatchFailure(op, "no TypeConverter attached");

    Type convertedResultTy =
        typeConverter->convertType(op.getResult().getType());
    auto memrefResultTy = dyn_cast<MemRefType>(convertedResultTy);
    if (!memrefResultTy)
      return rewriter.notifyMatchFailure(
          op, "apply result did not convert to MemRefType");

        // 在原 op 位置 alloc 出结果 buffer
    rewriter.setInsertionPoint(op);
    auto resultAlloc = rewriter.create<memref::AllocOp>(loc, memrefResultTy);

    SmallVector<scf::ForOp> loops;
    loops.reserve(rank);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      // 所有后续东西都插在 alloc 之后
      rewriter.setInsertionPointAfter(resultAlloc);

      // ==== 先在这个插入点上建每一维的 lb/ub/step constant ====
      SmallVector<Value> lbVals(rank), ubVals(rank), stepVals(rank);
      for (unsigned dim = 0; dim < rank; ++dim) {
        lbVals[dim] =
            rewriter.create<arith::ConstantIndexOp>(loc, lbs[dim]);
        ubVals[dim] =
            rewriter.create<arith::ConstantIndexOp>(loc, ubs[dim]);
        stepVals[dim] =
            rewriter.create<arith::ConstantIndexOp>(loc, 1);
      }

      // ==== 然后再建最外层 for ====
      auto outerFor = rewriter.create<scf::ForOp>(
          loc, lbVals[0], ubVals[0], stepVals[0],
          /*iterArgs=*/ValueRange());
      loops.push_back(outerFor);

      // ==== 再往里套 for（如果 rank > 1）====
      scf::ForOp current = outerFor;
      for (unsigned dim = 1; dim < rank; ++dim) {
        rewriter.setInsertionPointToStart(current.getBody());
        auto inner = rewriter.create<scf::ForOp>(
            loc, lbVals[dim], ubVals[dim], stepVals[dim],
            /*iterArgs=*/ValueRange());
        loops.push_back(inner);
        current = inner;
      }
    }

    scf::ForOp innermost = loops.back();


    // 准备把 apply 的区域内的 IR clone 到最里层 loop 里
    Block &bodyBlock = op.getBody().front();
    IRMapping mapping;

    // 1) region block args（索引）→ 对应维度的 iv
    for (unsigned dim = 0; dim < rank; ++dim)
      mapping.map(bodyBlock.getArgument(dim), loops[dim].getInductionVar());

    // 2) apply 的 temp 输入 → 转换后的 memref（adaptor 的 operands）
    auto origInputs = op.getInputs();
    auto convertedInputs = adaptor.getInputs();
    for (auto [oldV, newV] : llvm::zip(origInputs, convertedInputs))
      mapping.map(oldV, newV);

    // 在 innermost for 的 terminator 前插入
    Operation *loopTerm = innermost.getBody()->getTerminator();
    rewriter.setInsertionPoint(loopTerm);

    // clone apply body：遇到 access / yield 做 special handling
    for (Operation &nestedOp : bodyBlock.without_terminator()) {
      if (auto access = dyn_cast<AccessOp>(&nestedOp)) {
        // neptune_ir.access %tmp[<offsets>] : temp -> elem
        Value inputMemref = mapping.lookup(access.getInput());
        auto offsetsAttr = access.getOffsets();
        ArrayRef<int64_t> offs = offsetsAttr;

        SmallVector<Value> indices;
        indices.reserve(rank);
        for (unsigned dim = 0; dim < rank; ++dim) {
          Value iv = loops[dim].getInductionVar();
          int64_t off = offs[dim];
          if (off == 0) {
            indices.push_back(iv);
          } else {
            Value cOff = rewriter.create<arith::ConstantIndexOp>(loc, off);
            Value sum = rewriter.create<arith::AddIOp>(loc, iv, cOff);
            indices.push_back(sum);
          }
        }

        Value loaded =
            rewriter.create<memref::LoadOp>(loc, inputMemref, indices);
        mapping.map(access.getResult(), loaded);
        continue;
      }

      // 其它 op（一般是 arith / math 等标量运算）原样 clone
      Operation *cloned = rewriter.clone(nestedOp, mapping);
      for (auto [oldRes, newRes] :
           llvm::zip(nestedOp.getResults(), cloned->getResults()))
        mapping.map(oldRes, newRes);
    }

    // 处理 neptune_ir.yield：把标量写入 result memref
    auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());
    if (!yieldOp.getResults().empty()) {
      if (yieldOp.getResults().size() != 1)
        return yieldOp.emitOpError()
               << "only single-scalar yield is supported for now";

      Value yieldedOrig = yieldOp.getResults().front();
      Value yieldedVal = mapping.lookup(yieldedOrig);

      SmallVector<Value> resultIndices;
      resultIndices.reserve(rank);
      for (unsigned dim = 0; dim < rank; ++dim) {
        Value iv = loops[dim].getInductionVar();
        int64_t lb = lbs[dim];
        if (lb == 0) {
          resultIndices.push_back(iv);
        } else {
          Value cLb = rewriter.create<arith::ConstantIndexOp>(loc, lb);
          Value idx = rewriter.create<arith::SubIOp>(loc, iv, cLb);
          resultIndices.push_back(idx);
        }
      }

      rewriter.create<memref::StoreOp>(loc, yieldedVal, resultAlloc,
                                       resultIndices);
    }

    // apply 的 SSA result → 那块 result memref
    rewriter.replaceOp(op, resultAlloc.getResult());
    return success();
  }
};
} // namespace

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRSTENCILTOSCF
#include "Passes/NeptuneIRPasses.h.inc"

struct NeptuneIRStencilToSCFPass final
    : public impl::NeptuneIRStencilToSCFBase<NeptuneIRStencilToSCFPass> {
  void runOnOperation() override {
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *context = module.getContext();

    NeptuneIRTypeConverter typeConverter(context);
    ConversionTarget target(*context);

    // 保留这些 dialect
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // NeptuneIR dialect 最终要全部干掉
    target.addIllegalDialect<NeptuneIRDialect>();

    // func.func 需要做 signature 的类型转换（参数 / 返回值）
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      return typeConverter.isSignatureLegal(func.getFunctionType()) &&
             typeConverter.isLegal(&func.getBody());
    });

    // func.return：返回值类型必须合法（即已经不含 NeptuneIR::TempType/FieldType）
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp ret) {
      return typeConverter.isLegal(ret.getOperandTypes());
    });

    // 其它 op 如果所有 operand/result type 都已 legal，则可以保留
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return typeConverter.isLegal(op);
    });

    RewritePatternSet patterns(context);
    // func.func 的 signature conversion
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    // return 的 operand 也要跟着改
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    // 如果有 func.call 之类，也一起加：
    populateCallOpTypeConversionPattern(patterns, typeConverter);


    // NeptuneIR → memref + scf 的一系列 patterns
    patterns.add<WrapOpLowering, UnwrapOpLowering, LoadOpLowering,
                 StoreOpLowering, ApplyOpLowering>(typeConverter, context);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::Neptune::NeptuneIR
