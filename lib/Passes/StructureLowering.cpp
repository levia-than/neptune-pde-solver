#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRSTRUCTURELOWERINGPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;

//===----------------------------------------------------------------------===//
// 1) *_opdef -> func.func
//===----------------------------------------------------------------------===//

template <typename OpDefT>
struct OpDefToFuncLowering : OpRewritePattern<OpDefT> {
  using OpRewritePattern<OpDefT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpDefT op,
                                PatternRewriter &rewriter) const override {
    auto module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "must be inside a ModuleOp");

    Location loc = op.getLoc();

    // NeptuneIR opdef carries signature in function_type attr.
    auto funcTy = dyn_cast<FunctionType>(op.getFunctionType());
    if (!funcTy)
      return rewriter.notifyMatchFailure(op, "function_type must be FunctionType");

    StringRef name = op.getSymName();

    // Avoid clobbering an existing func with the same symbol.
    if (auto existing = module.template lookupSymbol<func::FuncOp>(name)) {
      // If the func already exists, treat this as an error: symbol collision.
      return rewriter.notifyMatchFailure(op, "func.func with same name already exists");
    }

    // Create func.func @name : (inputs)->(results)
    auto func = rewriter.create<func::FuncOp>(loc, name, funcTy);

    // Copy attrs over (except symbol + function_type, which are implied by func).
    // Keep any metadata like structure_key_hash, policy tags, etc.
    for (NamedAttribute na : op->getAttrs()) {
      StringRef attrName = na.getName().strref();
      if (attrName == SymbolTable::getSymbolAttrName())
        continue;
      if (attrName == "function_type")
        continue;
      func->setAttr(na.getName(), na.getValue());
    }

    // Move body region wholesale.
    // This avoids creating extra empty blocks in func.
    func.getBody().takeBody(op.getBody());

    // Rewrite terminators: neptune_ir.return -> func.return
    // (ReturnOp is your NeptuneIR terminator)
    SmallVector<ReturnOp> rets;
    func.walk([&](ReturnOp r) { rets.push_back(r); });

    for (ReturnOp r : rets) {
      rewriter.setInsertionPoint(r);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(r, r.getResults());
    }

    if (auto key = op->getAttr("structure_key"))
      func->setAttr("structure_key", key);
    
    if (auto hash = op->getAttr("structure_key_hash"))
      func->setAttr("structure_key_hash", hash);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 2) apply_* -> func.call
//===----------------------------------------------------------------------===//

template <typename ApplyT>
static LogicalResult lowerApplyToFuncCall(ApplyT op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();

  // Both apply_linear/apply_nonlinear should have:
  //   SymbolRefAttr $op
  //   Variadic<TempType> $inputs
  //   Variadic<TempType> $results
  //
  // ODS typically generates getOp() / getInputs().
  auto symRef = dyn_cast_or_null<SymbolRefAttr>(op.getOp());
  if (!symRef)
    return rewriter.notifyMatchFailure(op, "missing 'op' SymbolRefAttr");

  // func.call expects a flat symbol name; use the root reference.
  StringRef callee = symRef.getRootReference().getValue();

  auto call = rewriter.create<func::CallOp>(
      loc, callee, op.getResultTypes(), op.getInputs());

  // Preserve bounds metadata if present (apply_linear has it; nonlinear may or may not).
  if (Attribute b = op->getAttr("bounds"))
    call->setAttr("bounds", b);

  rewriter.replaceOp(op, call.getResults());
  return success();
}

struct ApplyLinearToCallLowering : OpRewritePattern<ApplyLinearOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ApplyLinearOp op,
                                PatternRewriter &rewriter) const override {
    return lowerApplyToFuncCall(op, rewriter);
  }
};

// If your op is named differently in generated C++, rename this type accordingly.
struct ApplyNonLinearToCallLowering : OpRewritePattern<ApplyNonLinearOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ApplyNonLinearOp op,
                                PatternRewriter &rewriter) const override {
    return lowerApplyToFuncCall(op, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct NeptuneIRStructureLoweringPass final
    : public mlir::Neptune::NeptuneIR::impl::NeptuneIRStructureLoweringPassBase<
          NeptuneIRStructureLoweringPass> {
  void runOnOperation() override {
    Operation *root = getOperation();
    auto module = dyn_cast<ModuleOp>(root);
    if (!module) {
      root->emitError() << "neptune-ir-structure-lowering expects to run on ModuleOp";
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = module.getContext();

    // Phase 1: lower opdefs to func.func first (so calls will resolve after phase 2).
    {
      RewritePatternSet patterns(ctx);
      patterns.add<OpDefToFuncLowering<LinearOpDefOp>,
                   OpDefToFuncLowering<NonlinearOpDefOp>>(ctx);

      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Phase 2: lower apply_* to func.call.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<ApplyLinearToCallLowering,
                   ApplyNonLinearToCallLowering>(ctx);

      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
