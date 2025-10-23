//===- LowerEvaluateToLoops.cpp - Lower neptune_ir.evaluate -> loops -----===//
//
// Minimal, robust lowering pass compatible with MLIR around LLVM/MLIR v21.
// - Uses IRMapping
// - Matches ops by name ("neptune_ir.evaluate", "neptune_ir.field.ref", "neptune_ir.field.add", ...)
// - Lowers expressions to memref.load / arith ops and memref.store inside nested scf.for
//
// NOTE: two project-specific hooks must be implemented for your repo:
//   * resolveStorageToMemRef(Value storageOperand) -> Value (memref) or null
//   * mapIndexOperandToLoopIV(Value indexOperand, ArrayRef<Value> ivs) -> Value (index for load)
//
// These are marked TODO in the code below and are deliberately left explicit so you
// can wire your symbol/descriptor scheme correctly.
//
//===----------------------------------------------------------------------===//

#include "Passes/LowerEvaluateToLoop.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

using namespace mlir;

namespace {
struct LowerEvaluateToLoopPass
    : public PassWrapper<LowerEvaluateToLoopPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  StringRef getArgument() const final { return "lower-evaluate-to-loop"; }
  StringRef getDescription() const final {
    return "Lower neptune_ir.evaluate (value-based) to scf.for + memref loads/stores.";
  }
};

/// Helper: create an index constant
static Value createIndexConst(OpBuilder &b, Location loc, int64_t x) {
  return b.create<arith::ConstantIndexOp>(loc, x);
}

/// PROJECT HOOK #1:
/// Resolve a 'storage operand' used by neptune_ir.field.ref into an actual memref Value
/// that can be loaded from. Return null Value() on failure.
///
/// Default fallback: if storageOperand is already MemRefType, return it.
/// Otherwise, user should fill a map from descriptor/decl op -> memref alloc value
/// and perform lookup here.
static Value resolveStorageToMemRef(Value storageOperand) {
  if (!storageOperand) return Value();
  if (isa<MemRefType>(storageOperand.getType())) return storageOperand;

  // TODO: your project's logic here.
  // Example patterns:
  //  - if storageOperand is result of a custom "storage.alloc" op, return its "alloc" value
  //  - if storageOperand is a descriptor Value, look up in a DenseMap<Operation*,Value> prepared earlier
  //
  // For now: return null to indicate not found.
  return Value();
}

/// PROJECT HOOK #2:
/// Map an index operand (from field.ref) to the actual loop induction variable / index value.
/// Strategy options (implement one that matches your IR):
///  - If indexOperand equals one of the loop ivs by SSA identity, return that iv.
///  - If indexOperand is a constant integer, return constant index.
///  - Else: choose a default mapping (e.g. use ivs[position]).
/// NOTE: This must be adapted to how your field.ref encodes indices.
/// `pos` is the ordinal (0-based) of the index among the field.ref indices.
static Value mapIndexOperandToLoopIV(Value indexOperand, ArrayRef<Value> ivs, OpBuilder &b,
                                     Location loc, unsigned pos) {
  // 1) identity match
  for (Value iv : ivs)
    if (iv == indexOperand)
      return iv;

  // 2) constant int
  if (auto def = indexOperand.getDefiningOp()) {
    if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      Attribute val = c.getValue();
      if (auto iAttr = dyn_cast<IntegerAttr>(val)) {
        return b.create<arith::ConstantIndexOp>(loc, iAttr.getInt());
      }
      if (auto fAttr = dyn_cast<FloatAttr>(val)) {
        // not index-like; skip
      }
    }
  }

  // 3) fallback: use ivs[pos] if present
  if (pos < ivs.size()) return ivs[pos];

  // 4) last resort: zero
  return b.create<arith::ConstantIndexOp>(loc, 0);
}

/// Recursively lower a NeptuneIR expression Value to a scalar SSA Value.
///
/// Recognized ops by name (string):
///  - "neptune_ir.field.ref" : loads from resolved memref using index operands
///  - "neptune_ir.field.add" / "field.sub" / "field.mul" / "field.div" : binary ops
///  - "neptune_ir.field.scale" : scale by attr "scalar" (FloatAttr) or second operand
///
/// Returns Value() on failure (caller should handle).
static Value lowerExprToScalar(Value expr, OpBuilder &b, Location loc, ArrayRef<Value> loopIvs) {
  if (!expr) return Value();
  // If it's already a scalar (no defining op), just return
  if (!expr.getDefiningOp()) return expr;

  Operation *op = expr.getDefiningOp();
  StringRef name = op->getName().getStringRef();

  // field.ref: operand0 = storage, remaining operands = indices
  if (name == "neptune_ir.field.ref") {
    if (op->getNumOperands() < 1) {
      op->emitError("field.ref expected at least storage operand");
      return Value();
    }
    Value storage = op->getOperand(0);
    Value mem = resolveStorageToMemRef(storage);
    if (!mem) {
      op->emitError("cannot resolve storage operand to memref in lowering");
      return Value();
    }

    // collect indices and map them to index values
    SmallVector<Value, 4> indices;
    for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i) {
      Value idxOp = op->getOperand(i);
      Value mapIdx = mapIndexOperandToLoopIV(idxOp, loopIvs, b, loc, i - 1);
      indices.push_back(mapIdx);
    }

    // memref.load
    return b.create<memref::LoadOp>(loc, mem, indices);
  }

  // binary ops
  if (name == "neptune_ir.field.add" || name == "neptune_ir.field.sub" ||
      name == "neptune_ir.field.mul" || name == "neptune_ir.field.div") {
    if (op->getNumOperands() < 2) {
      op->emitError("binary field op expects two operands");
      return Value();
    }
    Value lhs = lowerExprToScalar(op->getOperand(0), b, loc, loopIvs);
    Value rhs = lowerExprToScalar(op->getOperand(1), b, loc, loopIvs);
    if (!lhs || !rhs) return Value();

    Type ety = lhs.getType();
    if (isa<FloatType>(ety)) {
      if (name == "neptune_ir.field.add") return b.create<arith::AddFOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.sub") return b.create<arith::SubFOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.mul") return b.create<arith::MulFOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.div") return b.create<arith::DivFOp>(loc, lhs, rhs);
    } else {
      // integer ops - choose signed arithmetic by default
      if (name == "neptune_ir.field.add") return b.create<arith::AddIOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.sub") return b.create<arith::SubIOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.mul") return b.create<arith::MulIOp>(loc, lhs, rhs);
      if (name == "neptune_ir.field.div") return b.create<arith::DivSIOp>(loc, lhs, rhs);
    }
    return Value();
  }

  // field.scale: first operand expression, second optional scalar operand; or attr "scalar"
  if (name == "neptune_ir.field.scale") {
    if (op->getNumOperands() < 1) {
      op->emitError("field.scale expects at least one operand");
      return Value();
    }
    Value base = lowerExprToScalar(op->getOperand(0), b, loc, loopIvs);
    if (!base) return Value();

    // try attr "scalar"
    if (auto fAttr = op->getAttrOfType<FloatAttr>("scalar")) {
      Type ft = base.getType();
      if (auto fTy = dyn_cast<FloatType>(ft)) {
        // create float constant of appropriate type
        auto constAttr = FloatAttr::get(fTy, llvm::APFloat(fAttr.getValueAsDouble()));
        auto c = b.create<arith::ConstantOp>(loc, fTy, constAttr);
        return b.create<arith::MulFOp>(loc, base, c);
      }
    }

    // else if second operand exists, use it (assumed scalar)
    if (op->getNumOperands() > 1) {
      Value scalarV = op->getOperand(1);
      // if scalarV is not scalar, user must adjust upstream
      return b.create<arith::MulFOp>(loc, base, scalarV);
    }

    op->emitError("field.scale had no scalar attribute or operand");
    return Value();
  }

  // If it's an arithmetic constant op already, it will be returned by recursion above.
  // Unknown op: emit error
  op->emitError("unhandled op in NeptuneIR expr lowering: ") << name;
  return Value();
}

/// Build nested scf.for loops over memref shape and lower body to store scalar results.
static LogicalResult lowerEvaluateOp(Operation *evalOp) {
  // Expect two operands: dst memref and expr value
  if (evalOp->getNumOperands() < 2) {
    evalOp->emitError("evaluate expects dst memref and expr");
    return failure();
  }
  Value dst = evalOp->getOperand(0);
  Value expr = evalOp->getOperand(1);
  Location loc = evalOp->getLoc();

  auto dstMemTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstMemTy) {
    evalOp->emitError("evaluate dst must be a memref");
    return failure();
  }

  OpBuilder b(evalOp);
  // Prepare loop bounds (lb=0, ub = dimension or constant)
  unsigned rank = dstMemTy.getRank();
  SmallVector<Value, 4> lbs(rank), ubs(rank), steps(rank);
  for (unsigned d = 0; d < rank; ++d) {
    lbs[d] = createIndexConst(b, loc, 0);
    if (dstMemTy.isDynamicDim(d)) {
      ubs[d] = b.create<memref::DimOp>(loc, dst, d);
    } else {
      ubs[d] = createIndexConst(b, loc, dstMemTy.getShape()[d]);
    }
    steps[d] = createIndexConst(b, loc, 1);
  }

  // Build nested loops using scf.for ops: create outermost and nest inside
  // We create loops iteratively, collecting the induction variables.
  SmallVector<Value, 4> ivs; // induction variables
  SmallVector<scf::ForOp, 4> loops;
  Operation *insertionPt = evalOp; // insert loops before evalOp
  for (unsigned i = 0; i < rank; ++i) {
    b.setInsertionPoint(insertionPt);
    auto forOp = b.create<scf::ForOp>(loc, lbs[i], ubs[i], steps[i]);
    // move insertion point to start of forOp body
    Block *body = forOp.getBody();
    insertionPt = &body->back(); // at end; next loop will be inserted before this sentinel
    // the induction var (index) is in block arg 0 of body
    ivs.push_back(forOp.getInductionVar());
    loops.push_back(forOp);
    // next loop should be inserted at body->getTerminator() (which doesn't exist yet),
    // so set insertion to body->getTerminator() by setting insertionPt to terminator placeholder.
    insertionPt = forOp.getBody()->getTerminator(); // may be nullptr, but setInsertionPoint handles it
    // To be safe, set insertion point at start of body block
    b.setInsertionPointToStart(forOp.getBody());
  }

  // After creating all loops, we need to place body code in innermost loop
  scf::ForOp innerFor = loops.empty() ? scf::ForOp() : loops.back();
  if (loops.empty()) {
    // rank==0? (scalar memref) – directly compute and store
    b.setInsertionPoint(evalOp);
    Value scalar = lowerExprToScalar(expr, b, loc, ivs);
    if (!scalar) return failure();
    b.create<memref::StoreOp>(loc, scalar, dst, ArrayRef<Value>{});
  } else {
    // insert into the innermost for body
    b.setInsertionPointToStart(innerFor.getBody());

    // Lower expression using ivs mapping
    Value scalar = lowerExprToScalar(expr, b, loc, ivs);
    if (!scalar) return failure();

    // create store to dst at indices = ivs
    b.create<memref::StoreOp>(loc, scalar, dst, ivs);
  }

  // erase original evaluate op
  evalOp->erase();
  return success();
}

void LowerEvaluateToLoopPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Walk module and collect evaluate ops first (we cannot erase while iterating)
  SmallVector<Operation *, 8> evals;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "neptune_ir.evaluate") evals.push_back(op);
  });

  for (Operation *op : evals) {
    if (failed(lowerEvaluateOp(op))) {
      signalPassFailure();
      return;
    }
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::Neptune::NeptuneIR::createLowerEvaluateToLoopPass() {
  return std::make_unique<LowerEvaluateToLoopPass>();
}