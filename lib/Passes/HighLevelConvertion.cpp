#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRHIGHLEVELCONVERTIONPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

static Operation *lookupSym(Operation *anchor, SymbolRefAttr sym) {
  if (!sym)
    return nullptr;
  return SymbolTable::lookupNearestSymbolFrom(anchor, sym);
}

static bool isLinearOpDef(Operation *anchor, SymbolRefAttr sym) {
  if (auto *op = lookupSym(anchor, sym))
    return isa<LinearOpDefOp>(op);
  return false;
}

static bool isNonlinearOpDef(Operation *anchor, SymbolRefAttr sym) {
  if (auto *op = lookupSym(anchor, sym))
    return isa<NonlinearOpDefOp>(op);
  return false;
}

static BoundsAttr getBoundsFromState(Value state) {
  auto tmpTy = dyn_cast<TempType>(state.getType());
  if (!tmpTy)
    return {};
  return tmpTy.getBounds();
}

struct TimeAdvanceConvertion : mlir::OpRewritePattern<TimeAdvanceOp> {
  TimeAdvanceConvertion(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  mlir::LogicalResult
  matchAndRewrite(TimeAdvanceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value state = op.getState();
    Value dt = op.getDt();
    Type stTy = state.getType();

    TimeMethod method = op.getMethod();
    SymbolRefAttr rhsAttr = op.getRhsAttr();
    SymbolRefAttr systemAttr = op.getSystemAttr();
    SymbolRefAttr residualAttr = op.getResidualAttr();
    SymbolRefAttr jacAttr = op.getJacobianAttr();

    StringAttr solverAttr = op.getSolverAttr();
    FloatAttr tolAttr = op.getTolAttr();
    IntegerAttr maxItAttr = op.getMaxItersAttr();
    auto bounds = getBoundsFromState(state);

    switch (method) {
    case TimeMethod::kExplicit: {
      if (!rhsAttr)
        return rewriter.notifyMatchFailure(op, "explicit requires rhs");

      Value k;
      if (isLinearOpDef(op, rhsAttr)) {
        auto appl = rewriter.create<ApplyLinearOp>(
            loc, TypeRange{stTy}, rhsAttr, ValueRange{state},
            bounds ? bounds : BoundsAttr());
        k = appl.getResults().front();
      } else if (isNonlinearOpDef(op, rhsAttr)) {
        auto appn = rewriter.create<ApplyNonLinearOp>(
            loc, TypeRange{stTy}, rhsAttr, ValueRange{state},
            bounds ? bounds : BoundsAttr());
        k = appn.getResults().front();
      } else {
        return rewriter.notifyMatchFailure(
            op, "rhs must reference linear_opdef or nonlinear_opdef");
      }
      auto apply = rewriter.create<ApplyOp>(loc, stTy, ValueRange{state});

      Region &r = apply.getRegion();
      Block *body = new Block();
      r.push_back(body);
      body->addArgument(rewriter.getIndexType(), loc);

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(body);

      auto offs0 = rewriter.getDenseI64ArrayAttr({0});

      Value s0 =
          rewriter.create<AccessOp>(loc, rewriter.getF64Type(), state, offs0);
      Value k0 =
          rewriter.create<AccessOp>(loc, rewriter.getF64Type(), k, offs0);

      Value dt_k = rewriter.create<arith::MulFOp>(loc, dt, k0);
      Value out0 = rewriter.create<arith::AddFOp>(loc, s0, dt_k);

      rewriter.create<YieldOp>(loc, out0);

      rewriter.replaceOp(op, apply.getResult());
      return success();
    }
    case TimeMethod::kImplicitLinear: {
      if (!systemAttr)
        return rewriter.notifyMatchFailure(
            op, "implicit_linear requires rhs (linear_opdef)");

      if (!isLinearOpDef(op, systemAttr))
        return rewriter.notifyMatchFailure(
            op, "implicit_linear rhs must be linear_opdef");

      auto matTy = MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   rewriter.getF64Type());
      auto A = rewriter.create<AssembleMatrixOp>(
          loc, matTy, systemAttr, bounds ? bounds : BoundsAttr());

      auto sl = rewriter.create<SolveLinearOp>(
          loc, stTy, A.getResult(), state,
          op.getSolverAttr() ? op.getSolverAttr() : StringAttr(),
          op.getTolAttr() ? op.getTolAttr() : FloatAttr(),
          op.getMaxItersAttr() ? op.getMaxItersAttr() : IntegerAttr());

      rewriter.replaceOp(op, sl.getResult());
      return success();
    }
    case TimeMethod::kImplicitNonlinear: {
      if (!residualAttr)
        return rewriter.notifyMatchFailure(
            op, "implicit_nonlinear requires residual");

      // 这里把 time_advance 上的 solver/tol/max_iters 直接透传给
      // solve_nonlinear
      auto sn = rewriter.create<SolveNonlinearOp>(
          loc, stTy, residualAttr, jacAttr ? jacAttr : SymbolRefAttr(), state,
          /*captures=*/ValueRange{state}, // 你现在 time_advance 没有
                                     // captures，就先空
          op.getSolverAttr() ? op.getSolverAttr() : StringAttr(),
          op.getTolAttr() ? op.getTolAttr() : FloatAttr(),
          op.getMaxItersAttr() ? op.getMaxItersAttr() : IntegerAttr());

      rewriter.replaceOp(op, sn.getResult());
      return success();
    }
    case TimeMethod::kRuntime: {
      auto rt = rewriter.create<TimeAdvanceRuntimeOp>(
          loc, stTy, state, dt, op.getMethodAttr(), systemAttr, rhsAttr, residualAttr,
          jacAttr, op.getSolverAttr(), op.getTolAttr(), op.getMaxItersAttr());
      rewriter.replaceOp(op, rt.getResult());
      return success();
    }
    }

    // runtime: 留给你原来的 RuntimeLowering 去处理（直接 call runtime）
    return failure();
  }
};
} // namespace

namespace mlir::Neptune::NeptuneIR {
//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct NeptuneIRHighLevelConvertionPass final
    : public mlir::Neptune::NeptuneIR::impl::
          NeptuneIRHighLevelConvertionPassBase<
              NeptuneIRHighLevelConvertionPass> {
  void runOnOperation() override {
    Operation *root = getOperation();
    auto module = dyn_cast<ModuleOp>(root);

    MLIRContext *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<TimeAdvanceConvertion>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::Neptune::NeptuneIR
