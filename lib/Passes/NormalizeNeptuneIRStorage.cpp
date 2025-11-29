/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-10-20 20:15:10
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-28 16:45:32
 * @FilePath: /neptune-pde-solver/lib/Passes/NeptuneIRToMemRef.cpp
 * @Description:
 *  Current Pass try to convert all EvaluateOp to the real compute Op, e.g.
 *  Memref, Arith or other stuff.
 *  This would take EvaluateOp and its whole definition tree as input.
 * Copyright (c) 2025 by leviathan, All Rights Reserved.
 */

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::Neptune;

// type converter
namespace {

/// 把 BoundsAttr + elementType 变成一个静态 memref 的 shape。
static MemRefType getMemRefTypeFromBounds(Type elementType,
                                          NeptuneIR::BoundsAttr boundsAttr) {
  DenseI64ArrayAttr lbAttr = boundsAttr.getLb();
  DenseI64ArrayAttr ubAttr = boundsAttr.getUb();

  ArrayRef<int64_t> lbs = lbAttr.asArrayRef();
  ArrayRef<int64_t> ubs = ubAttr.asArrayRef();

  assert(lbs.size() == ubs.size() && "lb/ub rank mismatch in BoundsAttr");

  SmallVector<int64_t, 4> shape;
  shape.reserve(lbs.size());
  for (size_t i = 0; i < lbs.size(); ++i) {
    int64_t extent = ubs[i] - lbs[i];
    // 这里你可以额外做一些 sanity check，比如 extent > 0。
    shape.push_back(extent);
  }

  // 默认 memref layout/space，后面要支持 explicit layout/space 再扩展。
  return MemRefType::get(shape, elementType);
}

/// 把 Neptune 的 Field/Temp type 映射到 memref。
struct NeptuneIRTypeConverter : public TypeConverter {
  explicit NeptuneIRTypeConverter(MLIRContext *ctx) {
    // 对于非 NeptuneIR 类型，保持不变（identity）。
    addConversion([](Type ty) { return ty; });

    // FieldType / TempType → MemRefType
    addConversion([ctx](Type ty) -> Type {
      if (auto fieldTy = dyn_cast<NeptuneIR::FieldType>(ty)) {
        auto memrefTy = getMemRefTypeFromBounds(fieldTy.getElementType(),
                                                fieldTy.getBounds());
        return memrefTy;
      }
      if (auto tempTy = dyn_cast<NeptuneIR::TempType>(ty)) {
        auto memrefTy = getMemRefTypeFromBounds(tempTy.getElementType(),
                                                tempTy.getBounds());
        return memrefTy;
      }
      return ty;
    });

    // Source/Target materialization：统一用 builtin.unrealized_conversion_cast
    // 这是 MLIR 官方教程中推荐的 bridging 写法之一。
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return {};
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
  }
};

} // namespace

namespace {

/// neptune_ir.wrap %buffer : memref<...> -> !neptune_ir.field<...>
/// v0 lowering：直接把 Field 看成同一个 memref，wrap 变成 no-op。
struct WrapOpLowering : public OpConversionPattern<NeptuneIR::WrapOp> {
  using OpConversionPattern<NeptuneIR::WrapOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NeptuneIR::WrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 类型转换之后，operand/result 都已经是 memref<...> 了。
    rewriter.replaceOp(op, adaptor.getBuffer());
    return success();
  }
};

/// neptune_ir.unwrap %field : !neptune_ir.field<...> -> memref<...>
/// 同理变成 no-op。
struct UnwrapOpLowering : public OpConversionPattern<NeptuneIR::UnwrapOp> {
  using OpConversionPattern<NeptuneIR::UnwrapOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NeptuneIR::UnwrapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

/// neptune_ir.load %field : !neptune_ir.field<...> -> !neptune_ir.temp<...>
/// v0 设计里 Field/Temp 都映射成相同 shape 的 memref，所以这里也可以 no-op。
struct LoadOpLowering : public OpConversionPattern<NeptuneIR::LoadOp> {
  using OpConversionPattern<NeptuneIR::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NeptuneIR::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVarField());
    return success();
  }
};

/// neptune_ir.store %tmp to %field :
///   !neptune_ir.temp<...> to !neptune_ir.field<...>
/// v0：直接降成一个 memref.copy %tmp, %field。
/// （先忽略 bounds 的子域语义，后面可以用 subview + loop 细化。）
struct StoreOpLowering : public OpConversionPattern<NeptuneIR::StoreOp> {
  using OpConversionPattern<NeptuneIR::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NeptuneIR::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = adaptor.getValue();       // 源 temp (memref)
    Value varField = adaptor.getVarField(); // 目标 field (memref)

    rewriter.create<memref::CopyOp>(loc, value, varField);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NORMALIZENEPTUNEIRSTORAGE
#include "Passes/NeptuneIRPasses.h.inc"

struct NormalizeNeptuneIRStoragePass final
    : public impl::NormalizeNeptuneIRStorageBase<
          NormalizeNeptuneIRStoragePass> {
  void runOnOperation() override {
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = module->getContext();

    NeptuneIRTypeConverter typeConverter(ctx);

    RewritePatternSet patterns(ctx);
    patterns
        .add<WrapOpLowering, UnwrapOpLowering, LoadOpLowering, StoreOpLowering>(
            typeConverter, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalDialect<NeptuneIRDialect>(
      [](Operation *op) {
        return isa<ApplyOp, AccessOp, YieldOp>(op);
      });
    // Apply patterns greedily on the function body
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
