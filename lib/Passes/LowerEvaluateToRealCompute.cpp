/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-10-20 20:15:10
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-08 16:06:15
 * @FilePath: /neptune-pde-solver/lib/Passes/LowerEvaluateToRealCompute.cpp
 * @Description: 
 *  Current Pass try to convert all EvaluateOp to the real compute Op, e.g.
 *  Memref, Arith or other stuff.
 *  This would take EvaluateOp and its whole definition tree as input.
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
 */

#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Passes/NeptuneIRPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
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

namespace {
/// Utility: depth-limited DFS on Value->definingOp operands to find first
/// FieldRefOp. Avoid going into regions by only following op->getOperands().
static Operation *findFirstFieldRef(Value v, SmallPtrSetImpl<Operation *> &visited) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;
  if (!visited.insert(def).second)
    return nullptr;
  if (auto fr = dyn_cast<mlir::Neptune::NeptuneIR::FieldRefOp>(def))
    return def;
  for (Value opnd : def->getOperands()) {
    if (Operation *res = findFirstFieldRef(opnd, visited))
      return res;
  }
  return nullptr;
}

/// Collect enclosing scf.for ops from inside->outside.
static void collectEnclosingForOps(Operation *start, SmallVectorImpl<scf::ForOp> &outs) {
  Operation *cur = start->getParentOp();
  while (cur) {
    if (auto f = dyn_cast<scf::ForOp>(cur))
      outs.push_back(f);
    cur = cur->getParentOp();
  }
}


/// Try to ensure idxVal is available (dominates insertionPoint). If already
/// available, return it. Otherwise attempt to shallow-clone simple index ops
/// (arith.addi, arith.subi, arith.constant_index) into insertionPoint.
/// clonedCache prevents exponential cloning.
static std::optional<Value>
ensureIndexAvailable(Value idxVal, Operation *insertionPoint,
                     DominanceInfo *domInfo, OpBuilder &b,
                     DenseMap<Value, Value> &clonedCache) {
  if (!idxVal)
    return std::nullopt;

  // If cached
  if (clonedCache.count(idxVal))
    return clonedCache.lookup(idxVal);

  // If it's a block argument
  if (!idxVal.getDefiningOp()) {
    Block *argBlock = dyn_cast<BlockArgument>(idxVal).getOwner();
    if (!domInfo || domInfo->dominates(argBlock, insertionPoint->getBlock())) {
      clonedCache[idxVal] = idxVal;
      return idxVal;
    }
    return std::nullopt;
  }

  Operation *defOp = idxVal.getDefiningOp();
  // If definition already dominates insertion point -> reuse
  if (!domInfo || domInfo->dominates(defOp, insertionPoint)) {
    clonedCache[idxVal] = idxVal;
    return idxVal;
  }

  // Attempt shallow clone for simple ops
  // Only support arith::AddIOp, arith::SubIOp, arith::ConstantIndexOp for now.
  if (auto cst = dyn_cast<arith::ConstantIndexOp>(defOp)) {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertionPoint);
    auto nc = b.create<arith::ConstantIndexOp>(cst.getLoc(), cst.value());
    clonedCache[idxVal] = nc.getResult();
    return nc.getResult();
  }
  if (auto addi = dyn_cast<arith::AddIOp>(defOp)) {
    // ensure operands
    std::optional<Value> l = ensureIndexAvailable(addi.getLhs(), insertionPoint,
                                                  domInfo, b, clonedCache);
    std::optional<Value> r = ensureIndexAvailable(addi.getRhs(), insertionPoint,
                                                  domInfo, b, clonedCache);
    if (!l || !r)
      return std::nullopt;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertionPoint);
    auto na = b.create<arith::AddIOp>(addi.getLoc(), *l, *r);
    clonedCache[idxVal] = na.getResult();
    return na.getResult();
  }
  if (auto subi = dyn_cast<arith::SubIOp>(defOp)) {
    std::optional<Value> l = ensureIndexAvailable(subi.getLhs(), insertionPoint,
                                                  domInfo, b, clonedCache);
    std::optional<Value> r = ensureIndexAvailable(subi.getRhs(), insertionPoint,
                                                  domInfo, b, clonedCache);
    if (!l || !r)
      return std::nullopt;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertionPoint);
    auto ns = b.create<arith::SubIOp>(subi.getLoc(), *l, *r);
    clonedCache[idxVal] = ns.getResult();
    return ns.getResult();
  }

  // can't make available
  return std::nullopt;
}


/// Lower a Neptune IR expression value (value `v`) recursively to an SSA
/// scalar in the target dialect (arith/memref) using `rewriter`.
/// - `indices`: indices to use when encountering FieldRefOp (if the FieldRefOp
///   itself carried indices, you can still use them; this param is used as
///   fallback/replacement when the FieldRef is not directly present).
/// - `dom`: optional DominanceInfo to validate that any used index-def ops
///   dominate the evalOp. If nullptr, skip dominance checks.
/// Return Value() on failure.
static Value lowerExprValueToScalar(Value v, OpBuilder &b, Location loc,
                                    ArrayRef<Value> indices, DominanceInfo *dom) {
  // If v is produced by a NeptuneIR op, handle known ops
  if (Operation *def = v.getDefiningOp()) {
    // FieldRef -> memref.load
    if (auto fr = dyn_cast<mlir::Neptune::NeptuneIR::FieldRefOp>(def)) {
      Value storage = fr.getOperand(0); // first operand is storage (memref or descriptor)
      // The FieldRefOp may expose indices: fr.getIndices()
      SmallVector<Value, 4> frIdxs;
      for (Value iv : fr.getIndices()) frIdxs.push_back(iv);

      // If fr has explicit indices, use them; otherwise use provided fallback `indices`.
      SmallVector<Value, 4> useIdxs;
      if (!frIdxs.empty())
        useIdxs.assign(frIdxs.begin(), frIdxs.end());
      else
        useIdxs.assign(indices.begin(), indices.end());

      // Dominance check (simple): ensure defs of useIdxs dominate the load location.
      if (dom) {
        for (Value idxVal : useIdxs) {
          if (Operation *defOp = idxVal.getDefiningOp()) {
            // If defOp doesn't dominate the parent op of the load site, we cannot
            // reliably use it here. We'll still proceed (caller may check), but we
            // could also bail out.
            // (Here we do a soft check — for stricter behavior, return failure.)
            // e.g. if (!dom->dominates(defOp, def)) return Value();
          }
        }
      }

      // Create memref.load with useIdxs
      return b.create<memref::LoadOp>(loc, storage, useIdxs).getResult();
    }

    // FieldAdd: binary field add -> lower operands and arith.addf
    if (auto fa = dyn_cast<mlir::Neptune::NeptuneIR::FieldAddOp>(def)) {
      Value lhs = lowerExprValueToScalar(fa.getOperand(0), b, loc, indices, dom);
      Value rhs = lowerExprValueToScalar(fa.getOperand(1), b, loc, indices, dom);
      if (!lhs || !rhs) return Value();
      Type ft = lhs.getType();
      return b.create<arith::AddFOp>(loc, lhs, rhs).getResult();
    }

    // FieldSub -> arith.subf
    if (auto fs = dyn_cast<mlir::Neptune::NeptuneIR::FieldSubOp>(def)) {
      Value lhs = lowerExprValueToScalar(fs.getOperand(0), b, loc, indices, dom);
      Value rhs = lowerExprValueToScalar(fs.getOperand(1), b, loc, indices, dom);
      if (!lhs || !rhs) return Value();
      return b.create<arith::SubFOp>(loc, lhs, rhs).getResult();
    }

    // FieldMul / FieldDiv if present could be added similarly.

    // FieldScale: scale by scalar attr or operand
    if (auto fsc = dyn_cast<mlir::Neptune::NeptuneIR::FieldScaleOp>(def)) {
      // operand path
      Value in = lowerExprValueToScalar(fsc.getOperand(), b, loc, indices, dom);
      if (!in) return Value();

      // Try attribute named "scalar" first (TableGen used F64Attr:$scalar)
      if (auto fattr = def->getAttrOfType<FloatAttr>("scalar")) {
        // cast to element type of `in`
        Type elt = in.getType();
        if (!isa<FloatType>(elt)) {
          // If elt is index/other, skipping for now
          return Value();
        }
        FloatType fTy = cast<FloatType>(elt);
        // get double value
        double dv = fattr.getValueAsDouble();
        auto cst = b.create<arith::ConstantOp>(loc,
                       FloatAttr::get(fTy, APFloat(dv)));
        return b.create<arith::MulFOp>(loc, in, cst.getResult()).getResult();
      }
      // No scalar found
      return Value();
    }

    // If def is some other op we don't handle, try to recursively lower its operands
    // and if that produces something meaningful, use it. For unknown ops, fail.
    return Value();
  }

  // v is a block arg or something not defined by an op.
  // If it's from memref.load earlier or an SSA value forwarded in, just return v.
  return v;
}

/// Pattern that matches EvaluateOp and lowers it.
// 差不多就是给evaluateOp做lowering。
struct LowerEvaluatePattern
    : public OpRewritePattern<mlir::Neptune::NeptuneIR::EvaluateOp> {
  LowerEvaluatePattern(MLIRContext *ctx, DominanceInfo *dom,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<mlir::Neptune::NeptuneIR::EvaluateOp>(ctx, benefit),
        domInfo(dom) {}

  LogicalResult matchAndRewrite(mlir::Neptune::NeptuneIR::EvaluateOp evalOp,
                                PatternRewriter &rewriter) const override {
    Location loc = evalOp.getLoc();
    // dst memref must be operand 0
    Value dst = evalOp.getOperand(0);
    Value expr = evalOp.getOperand(1);

    auto dstMem = dyn_cast<MemRefType>(dst.getType());
    if (!dstMem) {
      evalOp.emitError("evaluate dst must be memref");
      return failure();
    }
    unsigned rank = dstMem.getRank();

    // 1) Try to find a FieldRefOp within the expr tree (operand DFS)
    SmallPtrSet<Operation *, 8> visited;
    Operation *foundFrOp = findFirstFieldRef(expr, visited);

    SmallVector<Value, 4> indices;
    if (foundFrOp) {
      auto fr = cast<mlir::Neptune::NeptuneIR::FieldRefOp>(foundFrOp);
      for (Value v : fr.getIndices())
        indices.push_back(v);
      // If fieldRef has fewer indices than memref rank, we'll try to fill using enclosing fors
    }

    // 2) If we still don't have enough indices, try enclosing scf.for
    if (indices.size() < rank) {
      SmallVector<scf::ForOp, 4> enclosing;
      collectEnclosingForOps(evalOp, enclosing); // inner->outer
      for (auto &f : enclosing) {
        if (indices.size() >= rank) break;
        indices.push_back(f.getInductionVar());
      }
      // keep only the most-inner N indices (if more than rank collected)
      if (indices.size() > rank)
        indices.resize(rank);
    }

    if (indices.size() != rank) {
      if (rank != 0) {
        evalOp.emitError("evaluate with no field.ref cannot target non-zero-rank memref");
        return failure();
      }
    }

    // 3) Lower expression to scalar
    // We call lowerExprValueToScalar on expr; it will recursively lower NeptuneIR ops.
    Value scalar = lowerExprValueToScalar(expr, rewriter, loc, indices, domInfo);
    if (!scalar) {
      evalOp.emitError("failed lowering expression to scalar");
      return failure();
    }

    // 4) Insert memref.store
    if (rank == 0) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(evalOp, scalar, dst, ValueRange{});
    } else {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(evalOp, scalar, dst, indices);
    }

    return success();
  }

  DominanceInfo *domInfo;
};
} // namespace

namespace mlir::Neptune::NeptuneIR {
#define GEN_PASS_DEF_LOWEREVALUATETOREALCOMPUTE
#include "Passes/NeptuneIRPasses.h.inc"

struct LowerEvaluateToRealComputePass final
    : public impl::LowerEvaluateToRealComputeBase<
          LowerEvaluateToRealComputePass> {
  void runOnOperation() override {
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    for (auto func : module.getOps<func::FuncOp>()) {
      DominanceInfo dom(func);
      RewritePatternSet patterns(&getContext());
      patterns.add<LowerEvaluatePattern>(&getContext(), &dom);

      // Apply patterns greedily on the function body
      if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace mlir::Neptune::NeptuneIR
