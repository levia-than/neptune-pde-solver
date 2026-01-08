#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MD5.h"

using namespace mlir;

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_NEPTUNEIRVERIFYANNOTATEPASS
#include "Passes/NeptuneIRPasses.h.inc"
} // namespace mlir::Neptune::NeptuneIR

namespace {
using namespace mlir::Neptune::NeptuneIR;
namespace arith = mlir::arith;

//===----------------------------------------------------------------------===//
// Helpers: hash + serialization
//===----------------------------------------------------------------------===//

static uint64_t hashString(StringRef s) {
  llvm::MD5 md5;
  md5.update(s);
  llvm::MD5::MD5Result r;
  md5.final(r);
  return r.low();
}

static std::string serializeBounds(BoundsAttr b) {
  if (!b) return "none";
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "lb=";
  llvm::interleaveComma(b.getLb().asArrayRef(), os);
  os << ";ub=";
  llvm::interleaveComma(b.getUb().asArrayRef(), os);
  return os.str();
}

static std::string serializeOffsets(DenseI64ArrayAttr a) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  llvm::interleaveComma(a.asArrayRef(), os);
  os << "]";
  return os.str();
}

static std::string printType(Type t) {
  std::string out;
  llvm::raw_string_ostream os(out);
  t.print(os);
  return os.str();
}

static bool isArithConstant(Value v) {
  if (!v) return false;
  if (v.getDefiningOp<arith::ConstantOp>()) return true;
  if (v.getDefiningOp<arith::ConstantIndexOp>()) return true;
  if (v.getDefiningOp<arith::ConstantIntOp>()) return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Apply-like region checks (supports neptune_ir.apply + apply_nonlinear by name)
// We treat anything with:
//  - attr "bounds": BoundsAttr
//  - optional attr "shape"
//  - 1 region (body)
//  - terminator neptune_ir.yield
// as "apply-like".
//===----------------------------------------------------------------------===//

static LogicalResult checkApplyLike(Operation *op, bool requireLinearBody) {
    auto emit = [&](Twine msg) { return op->emitOpError(msg); };

  auto bounds = op->getAttrOfType<BoundsAttr>("bounds");
  if (!bounds) return emit("missing required 'bounds' attribute");

  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  if (lb.size() != ub.size())
    return emit("bounds lb/ub rank mismatch");

  unsigned rank = (unsigned)lb.size();
  unsigned numInputs = op->getNumOperands();

  if (op->getNumRegions() != 1)
    return emit("apply-like op must have exactly 1 region");

  Region &r = op->getRegion(0);
  if (!llvm::hasSingleElement(r))
    return emit("apply-like region must have a single block");

  Block &b = r.front();

  // NEW: [rank x index] + [numInputs x operand types]
  if (b.getNumArguments() != rank + numInputs)
    return emit(Twine("apply-like region block arg count must be (bounds rank + number of inputs) = ")
                + Twine(rank + numInputs) + ", but got " + Twine(b.getNumArguments()));

  for (unsigned d = 0; d < rank; ++d) {
    if (!b.getArgument(d).getType().isIndex())
      return emit(Twine("apply-like region block argument #") + Twine(d) + " must be index type");
  }

  for (unsigned i = 0; i < numInputs; ++i) {
    std::string expectTypeStr;
    std::string gotTypeStr;
    llvm::raw_string_ostream osForExpectTypeStr(expectTypeStr);
    llvm::raw_string_ostream osForGotTypeStr(gotTypeStr);
    op->getOperand(i).getType().print(osForExpectTypeStr);
    b.getArgument(rank + i).getType().print(osForGotTypeStr);
    if (gotTypeStr != expectTypeStr)
      return emit(Twine("apply-like region input arg #") + Twine(rank + i) +
                  " type mismatch: expect " + expectTypeStr + " but got " + gotTypeStr);
  }

  Operation *term = b.getTerminator();
  auto yield = dyn_cast<YieldOp>(term);
  if (!yield)
    return emit("apply-like region must terminate with neptune_ir.yield");

  if (yield.getNumOperands() != 1)
    return yield.emitOpError("MVP: only single-scalar yield is supported");

  if (op->getNumResults() == 1) {
    if (auto outTemp = dyn_cast<TempType>(op->getResult(0).getType())) {
      Type elem = outTemp.getElementType();
      if (yield.getOperand(0).getType() != elem)
        return yield.emitOpError("yield operand type must equal apply result element type");
    }
  }
  for (Operation &inner : b.getOperations()) {
    if (&inner == term) continue;

    if (auto acc = dyn_cast<AccessOp>(inner)) {
      auto inTemp = dyn_cast<TempType>(acc.getInput().getType());
      if (!inTemp)
        return acc.emitOpError("access input must be TempType");
      auto offs = acc.getOffsetsAttr().asArrayRef();
      auto inLb = inTemp.getBounds().getLb().asArrayRef();
      if (offs.size() != lb.size() || inLb.size() != lb.size())
        return acc.emitOpError("offsets rank must match apply bounds rank");

      // Stronger type check: access result == elementType
      if (acc.getResult().getType() != inTemp.getElementType())
        return acc.emitOpError("result type must equal input Temp element type");

      continue;
    }

    // For linear regions: restrict to affine-linear scalar algebra
    if (requireLinearBody) {
      bool ok =
          isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp>(inner) ||
          isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp>(inner) ||
          isa<arith::NegFOp>(inner) ||
          isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncFOp, arith::TruncIOp>(inner) ||
          isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(inner) ||
          isa<arith::MulFOp, arith::MulIOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(inner);

      if (!ok) {
        return inner.emitOpError("operation not allowed inside linear apply-like region");
      }

      // Enforce Mul/Div are linear: one operand must be constant (Div: rhs must be constant)
      if (auto mulF = dyn_cast<arith::MulFOp>(inner)) {
        if (!(isArithConstant(mulF.getLhs()) || isArithConstant(mulF.getRhs())))
          return mulF.emitOpError("MulFOp in linear region must multiply by a constant");
      }
      if (auto mulI = dyn_cast<arith::MulIOp>(inner)) {
        if (!(isArithConstant(mulI.getLhs()) || isArithConstant(mulI.getRhs())))
          return mulI.emitOpError("MulIOp in linear region must multiply by a constant");
      }
      if (auto divF = dyn_cast<arith::DivFOp>(inner)) {
        if (!isArithConstant(divF.getRhs()))
          return divF.emitOpError("DivFOp in linear region must divide by a constant");
      }
      if (auto divSI = dyn_cast<arith::DivSIOp>(inner)) {
        if (!isArithConstant(divSI.getRhs()))
          return divSI.emitOpError("DivSIOp in linear region must divide by a constant");
      }
      if (auto divUI = dyn_cast<arith::DivUIOp>(inner)) {
        if (!isArithConstant(divUI.getRhs()))
          return divUI.emitOpError("DivUIOp in linear region must divide by a constant");
      }

      continue;
    }

    // Nonlinear apply-like: we only disallow stray NeptuneIR ops except access/yield
    if (inner.getDialect() &&
        inner.getDialect()->getNamespace() == NeptuneIRDialect::getDialectNamespace()) {
      return inner.emitOpError("unexpected NeptuneIR op inside nonlinear apply-like region (only access/yield allowed)");
    }
  }

  return success();
}

static bool isApplyLike(Operation *op) {
  // canonical apply
  if (isa<ApplyOp>(op)) return true;

  // tolerate your possible new op name without hard-binding to its C++ class
  StringRef name = op->getName().getStringRef();
  if (name == "neptune_ir.apply_nonlinear") return true;
  if (name == "neptune_ir.apply_non_linear") return true;
  return false;
}

// Build a “structure_key” for an opdef by scanning apply-like regions.
static void buildAndAttachStructureKey(Operation *opdef,
                                       bool isLinear,
                                       PatternRewriter *maybeRewriter = nullptr) {
  auto *ctx = opdef->getContext();

  SmallVector<std::string> boundsParts;
  SmallVector<std::string> shapes;
  SmallVector<std::string> offsets;
  SmallVector<std::string> scalarOps; // extra for nonlinear uniqueness

  opdef->walk([&](Operation *op) {
    if (!isApplyLike(op)) return;

    if (auto b = op->getAttrOfType<BoundsAttr>("bounds"))
      boundsParts.push_back(serializeBounds(b));

    if (auto s = op->getAttr("shape")) {
      std::string ss;
      llvm::raw_string_ostream os(ss);
      os << s;
      shapes.push_back(os.str());
    }

    // collect access offsets + scalar op names
    Region &r = op->getRegion(0);
    Block &blk = r.front();
    Operation *term = blk.getTerminator();

    for (Operation &inner : blk.getOperations()) {
      if (&inner == term) continue;
      if (auto acc = dyn_cast<AccessOp>(inner)) {
        offsets.push_back(serializeOffsets(acc.getOffsetsAttr()));
        continue;
      }
      // for nonlinear opdef, fold scalar op names into key to avoid “same offsets but different math”
      if (!isLinear) {
        scalarOps.push_back(inner.getName().getStringRef().str());
      }
    }
  });

  llvm::sort(boundsParts);
  llvm::sort(shapes);
  llvm::sort(offsets);
  llvm::sort(scalarOps);

  std::string sig;
  if (auto fnTyAttr = opdef->getAttrOfType<TypeAttr>("function_type")) {
    sig = printType(fnTyAttr.getValue());
  }

  std::string key =
      llvm::formatv("kind={0}|sig={1}|b:{2}|s:{3}|o:{4}|ops:{5}",
                    isLinear ? "linear" : "nonlinear",
                    sig,
                    llvm::join(boundsParts, ";"),
                    llvm::join(shapes, ";"),
                    llvm::join(offsets, ";"),
                    llvm::join(scalarOps, ";"))
          .str();

  uint64_t h = hashString(key);

  opdef->setAttr("structure_key", StringAttr::get(ctx, key));
  opdef->setAttr("structure_key_hash",
                 IntegerAttr::get(IntegerType::get(ctx, 64), h));
}

//===----------------------------------------------------------------------===//
// Pass: Verify + Annotate
//===----------------------------------------------------------------------===//

struct NeptuneIRVerifyAnnotatePass final
    : public mlir::Neptune::NeptuneIR::impl::NeptuneIRVerifyAnnotatePassBase<
          NeptuneIRVerifyAnnotatePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTable(module);

    bool hadError = false;

    auto verifyOpDefCommon = [&](Operation *opdef, bool isLinear) -> LogicalResult {
      // function_type must exist + be FunctionType
      auto fnTyAttr = opdef->getAttrOfType<TypeAttr>("function_type");
      if (!fnTyAttr) return opdef->emitOpError("missing required 'function_type' attribute");
      auto fnTy = dyn_cast<FunctionType>(fnTyAttr.getValue());
      if (!fnTy) return opdef->emitOpError("'function_type' must be a FunctionType");

      // signature must be (Temp...) -> (Temp...)
      for (Type t : fnTy.getInputs())
        if (!isa<TempType>(t))
          return opdef->emitOpError("opdef inputs must be TempType");
      for (Type t : fnTy.getResults())
        if (!isa<TempType>(t))
          return opdef->emitOpError("opdef results must be TempType");

      // body: single region, single block
      if (opdef->getNumRegions() != 1)
        return opdef->emitOpError("expects exactly 1 region body");
      Region &body = opdef->getRegion(0);
      if (!llvm::hasSingleElement(body))
        return opdef->emitOpError("expects single-block body");

      Block &blk = body.front();
      auto *term = blk.getTerminator();
      auto ret = dyn_cast<ReturnOp>(term);
      if (!ret)
        return opdef->emitOpError("body must terminate with neptune_ir.return");

      if (ret.getNumOperands() != fnTy.getNumResults())
        return ret.emitOpError("return operand count must match function results");

      for (auto it : llvm::zip(ret.getOperands(), fnTy.getResults())) {
        if (std::get<0>(it).getType() != std::get<1>(it))
          return ret.emitOpError("return operand types must match function results");
      }

      // verify apply-like ops inside body
      opdef->walk([&](Operation *inner) {
        if (!isApplyLike(inner)) return;
        if (failed(checkApplyLike(inner, /*requireLinearBody=*/isLinear))) {
          hadError = true;
        }
      });

      return success();
    };

    // Verify linear_opdef
    for (auto def : module.getOps<LinearOpDefOp>()) {
      if (failed(verifyOpDefCommon(def.getOperation(), /*isLinear=*/true))) {
        hadError = true;
        continue;
      }
      buildAndAttachStructureKey(def.getOperation(), /*isLinear=*/true);
    }

    // Verify nonlinear_opdef (your new op)
    for (auto def : module.getOps<NonlinearOpDefOp>()) {
      if (failed(verifyOpDefCommon(def.getOperation(), /*isLinear=*/false))) {
        hadError = true;
        continue;
      }
      buildAndAttachStructureKey(def.getOperation(), /*isLinear=*/false);
    }

    // Verify symbol references resolve
    module.walk([&](Operation *op) {
      for (auto namedAttr : op->getAttrs()) {
        if (auto sym = dyn_cast<SymbolRefAttr>(namedAttr.getValue())) {
          if (!SymbolTable::lookupNearestSymbolFrom(op, sym)) {
            op->emitError() << "unresolved symbol reference " << sym;
            hadError = true;
          }
        }
      }
    });

    if (hadError)
      signalPassFailure();
  }
};

} // namespace
