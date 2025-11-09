//===- SymbolicSimplify.cpp - NeptuneIR symbolic simplification ----------===//
//
// Minimal SymbolicSimplify pass for NeptuneIR.
//
//===----------------------------------------------------------------------===//

#include "Passes/NeptuneIRPasses.h"

#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

/// Fold nested scales: neptune_ir.field.scale(neptune_ir.field.scale(x, s1),
/// s2)
///   => neptune_ir.field.scale(x, s1*s2)
struct FoldNestedFieldScale : public OpRewritePattern<Neptune::NeptuneIR::FieldScaleOp> {
  using OpRewritePattern<Neptune::NeptuneIR::FieldScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Neptune::NeptuneIR::FieldScaleOp op,
                                PatternRewriter &rewriter) const override {
    // require scalar attribute on this op
    auto scalarAttr = op.getScalarAttr();
    if (!scalarAttr || !isa<FloatAttr>(scalarAttr))
      return failure();

    // LHS must be another FieldScaleOp with a FloatAttr
    Value lhs = op.getLhs();
    auto innerScale = lhs.getDefiningOp<Neptune::NeptuneIR::FieldScaleOp>();
    if (!innerScale)
      return failure();
    auto innerAttr = innerScale.getScalarAttr();
    if (!innerAttr || !isa<FloatAttr>(innerAttr))
      return failure();

    // Multiply the two float attrs (double precision)
    double s1 = cast<FloatAttr>(innerAttr).getValueAsDouble();
    double s2 = cast<FloatAttr>(scalarAttr).getValueAsDouble();
    double prod = s1 * s2;

    // create a new FieldScaleOp with combined scalar attr
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    auto ctx = rewriter.getContext();
    auto newAttr = FloatAttr::get(rewriter.getF64Type(), APFloat(prod));

    // replace op with FieldScale(returnType, innerScale.lhs, newAttr)
    rewriter.replaceOpWithNewOp<Neptune::NeptuneIR::FieldScaleOp>(op, op->getResultTypes(), innerScale.getLhs(), newAttr);
    return success();
  }
};

/// Elide scale by 1.0: neptune_ir.field.scale(x, 1.0) => x
struct ElideScaleOne : public OpRewritePattern<Neptune::NeptuneIR::FieldScaleOp> {
  using OpRewritePattern<Neptune::NeptuneIR::FieldScaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Neptune::NeptuneIR::FieldScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto scalarAttr = op.getScalarAttr();
    if (!scalarAttr || !isa<FloatAttr>(scalarAttr))
      return failure();
    double s = cast<FloatAttr>(scalarAttr).getValueAsDouble();
    // treat 1.0 as identity (use exact compare; if you want tolerance use fabs)
    if (s == 1.0) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};

/// Optional: fold add(x, x) -> scale(x, 2.0)
struct FoldAddSame : public OpRewritePattern<Neptune::NeptuneIR::FieldAddOp> {
  using OpRewritePattern<Neptune::NeptuneIR::FieldAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Neptune::NeptuneIR::FieldAddOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op.getLhs();
    Value b = op.getRhs();
    if (a != b)
      return failure();
    // create FieldScale(a, 2.0)
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    auto newAttr = FloatAttr::get(rewriter.getF64Type(), APFloat(2.0));
    rewriter.replaceOpWithNewOp<Neptune::NeptuneIR::FieldScaleOp>(op, op->getResultTypes(), a, newAttr);
    return success();
  }
};

} // end anonymous namespace

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_SYMBOLICSIMPLIFY
#include "Passes/NeptuneIRPasses.h.inc"

struct SymbolicSimplifyPass final
    : public impl::SymbolicSimplifyBase<SymbolicSimplifyPass> {
  void runOnOperation() override {
    ModuleOp module = llvm::dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldNestedFieldScale, ElideScaleOne, FoldAddSame>(ctx);

    // You may want to run a few iterations (patterns are greedy and internal)
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      module.emitError("SymbolicSimplify: pattern application failed");
      signalPassFailure();
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR