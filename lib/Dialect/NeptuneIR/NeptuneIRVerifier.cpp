/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-11-29 11:41:01
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-29 11:48:09
 * @FilePath: /neptune-pde-solver/lib/Dialect/NeptuneIR/NeptuneIRVerifier.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by leviathan, All Rights Reserved. 
 */
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

namespace {
/// Conservative linearity check for linear_opdef:
/// - single-block body
/// - entry args match function type inputs
/// - terminator is neptune_ir.return with types matching function results
/// - operations restricted to a linear / analyzable subset
LogicalResult verifyLinearApplyRegion(ApplyOp apply) {
  if (!llvm::hasSingleElement(apply.getBody()))
    return apply.emitOpError("apply region must have a single block");
  Block &block = apply.getBody().front();
  auto *terminator = block.getTerminator();
  auto yield = dyn_cast<YieldOp>(terminator);
  if (!yield)
    return apply.emitOpError("apply region must terminate with neptune_ir.yield");

  auto isAllowed = [](Operation *inner) {
    return isa<AccessOp, YieldOp>(inner) ||
           isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
               arith::MulFOp, arith::ConstantOp>(inner);
  };
  for (Operation &inner : block.getOperations()) {
    if (&inner == terminator)
      continue;
    if (!isAllowed(&inner))
      return inner.emitOpError("op not allowed inside apply for linear_opdef");
  }
  return success();
}

LogicalResult verifyLinearOpBody(LinearOpDefOp op, FunctionType funcTy) {
  if (!llvm::hasSingleElement(op.getBody()))
    return op.emitOpError("expects a single-block body");

  Block &block = op.getBody().front();
  if (block.getNumArguments() != funcTy.getNumInputs())
    return op.emitOpError("block arg count must match function inputs");

  for (auto it : llvm::zip(block.getArguments(), funcTy.getInputs())) {
    if (std::get<0>(it).getType() != std::get<1>(it))
      return op.emitOpError("block argument types must match function inputs");
  }

  auto *terminator = block.getTerminator();
  auto ret = dyn_cast<ReturnOp>(terminator);
  if (!ret)
    return op.emitOpError("body must terminate with neptune_ir.return");
  if (ret.getNumOperands() != funcTy.getNumResults())
    return op.emitOpError("return operand count must match function results");
  for (auto it : llvm::zip(ret.getOperands(), funcTy.getResults())) {
    if (std::get<0>(it).getType() != std::get<1>(it))
      return op.emitOpError("return operand types must match function results");
  }

  auto isAllowedOp = [](Operation *inner) {
    return isa<AccessOp, ApplyOp, ApplyLinearOp, YieldOp, ReduceOp,
               AsTensorOp, FromTensorOp, ReturnOp>(inner) ||
           isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
               arith::MulFOp, arith::ConstantOp>(inner);
  };

  for (Operation &inner : block.getOperations()) {
    if (&inner == terminator)
      continue;

    if (inner.getNumRegions() > 0 && !isa<ApplyOp>(&inner))
      return inner.emitOpError(
          "only neptune_ir.apply may contain regions in linear_opdef");

    if (!isAllowedOp(&inner))
      return inner.emitOpError("operation not allowed in linear_opdef body");

    if (auto apply = dyn_cast<ApplyOp>(&inner)) {
      if (failed(verifyLinearApplyRegion(apply)))
        return failure();
    }

    if (auto iface = dyn_cast<MemoryEffectOpInterface>(&inner)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      bool hasWrite = llvm::any_of(
          effects, [](const MemoryEffects::EffectInstance &eff) {
            return isa<MemoryEffects::Write, MemoryEffects::Allocate>(
                eff.getEffect());
          });
      if (hasWrite)
        return inner.emitOpError(
            "linear_opdef body must be free of writes/allocations");
    }
  }
  return success();
}
} // namespace

LogicalResult WrapOp::verify() {
  // TODO: 检查 memref elementType == field.element
  return success();
}

LogicalResult UnwrapOp::verify() {
  // TODO: 检查 field.element == memref.element
  return success();
}

LogicalResult LoadOp::verify() {
  // TODO: 检查 field / temp 的 elementType、bounds/location 一致
  return success();
}

LogicalResult AccessOp::verify() {
  // TODO: 检查 offsets 维度是否匹配 temp 的维度等
  return success();
}

LogicalResult ApplyOp::verify() {
  auto bounds = getBounds();
  unsigned rank = bounds.getLb().size();
  if (rank == 0)
    return emitOpError("0-D apply not supported");

  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // 新规则：rank 个 index + numInputs 个 temp
  if (body.getNumArguments() != rank + numInputs)
    return emitOpError("apply-like region block arg count must be (bounds rank + number of inputs) = ")
           << (rank + numInputs) << ", but got " << body.getNumArguments();

  // 前 rank 个必须是 index
  for (unsigned d = 0; d < rank; ++d) {
    if (!body.getArgument(d).getType().isIndex())
      return emitOpError("region arg #") << d << " must be index";
  }

  // 后 numInputs 个必须与 operands 的类型一一对应
  for (unsigned i = 0; i < numInputs; ++i) {
    Type expect = getInputs()[i].getType();
    Type got = body.getArgument(rank + i).getType();
    if (got != expect)
      return emitOpError("region input arg #") << (rank + i)
             << " type mismatch: expect " << expect << " but got " << got;
  }

  return success();
}

LogicalResult StoreOp::verify() {
  // TODO: 检查 value/temp 与目标 field elementType & bounds 兼容
  return success();
}

LogicalResult ReduceOp::verify() {
  return success();
}

LogicalResult LinearOpDefOp::verify() {
  auto funcTy = dyn_cast<FunctionType>(getFunctionType());
  if (!funcTy)
    return emitOpError("function_type must be a FunctionType");
  for (Type ty : funcTy.getInputs()) {
    if (!isa<TempType>(ty))
      return emitOpError("inputs of linear_opdef must be TempType");
  }
  for (Type ty : funcTy.getResults()) {
    if (!isa<TempType>(ty))
      return emitOpError("results of linear_opdef must be TempType");
  }
  if (failed(verifyLinearOpBody(*this, funcTy)))
    return failure();
  return success();
}

LogicalResult ApplyLinearOp::verify() {
  return success();
}

LogicalResult AsTensorOp::verify() {
  return success();
}

LogicalResult FromTensorOp::verify() {
  return success();
}

LogicalResult AssembleMatrixOp::verify() {
  auto memrefTy = dyn_cast<MemRefType>(getMatrix().getType());
  if (!memrefTy)
    return emitOpError("result must be a memref");

  if (memrefTy.getRank() != 2)
    return emitOpError("result memref must be rank-2");

  if (!memrefTy.isDynamicDim(0) || !memrefTy.isDynamicDim(1))
    return emitOpError("result memref must have dynamic dims (?x?)");

  auto elemTy = dyn_cast<FloatType>(memrefTy.getElementType());
  if (!elemTy || elemTy.getWidth() != 64)
    return emitOpError("result element type must be f64 (MVP)");

  // 2) resolve symbol
  auto sym = getOp();
  if (!sym)
    return emitOpError("requires symbol reference");

  Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, sym);
  if (!target)
    return emitOpError("op must reference an existing symbol");

  // 3) accept linear_opdef OR func.func (post-structure-lowering)
  FunctionType funcTy;
  if (auto def = dyn_cast<LinearOpDefOp>(target)) {
    funcTy = dyn_cast<FunctionType>(def.getFunctionType());
  } else if (auto fn = dyn_cast<func::FuncOp>(target)) {
    funcTy = fn.getFunctionType();
  } else if (auto fnIfc = dyn_cast<FunctionOpInterface>(target)) {
    funcTy = dyn_cast<FunctionType>(fnIfc.getFunctionType());
  } else {
    return emitOpError("op must reference a callable symbol (linear_opdef or func.func)");
  }

  if (!funcTy)
    return emitOpError("referenced symbol must have a FunctionType");

  // 4) enforce signature element type is f64 (不要放过 i32 这种情况)
  auto checkArgOrRes = [&](Type ty, StringRef what) -> LogicalResult {
    if (auto t = dyn_cast<TempType>(ty)) {
      auto et = dyn_cast<FloatType>(t.getElementType());
      if (!et || et.getWidth() != 64)
        return emitOpError() << what << " element type must be f64 for assemble_matrix MVP";
      return success();
    }

    // 如果你未来允许在 dataflow 后再 assemble，可以加 memref 路径：
    if (auto mr = dyn_cast<MemRefType>(ty)) {
      auto et = dyn_cast<FloatType>(mr.getElementType());
      if (!et || et.getWidth() != 64)
        return emitOpError() << what << " element type must be f64 for assemble_matrix MVP";
      return success();
    }

    return emitOpError() << what << " must be TempType (or memref after lowering) for assemble_matrix MVP";
  };

  for (Type ty : funcTy.getInputs())
    if (failed(checkArgOrRes(ty, "operator input")))
      return failure();

  for (Type ty : funcTy.getResults())
    if (failed(checkArgOrRes(ty, "operator result")))
      return failure();

  return success();
}

LogicalResult SolveLinearOp::verify() {
  auto memrefTy = dyn_cast<MemRefType>(getSystem().getType());
  if (!memrefTy)
    return emitOpError("system must be a memref");
  if (!memrefTy.hasRank() || memrefTy.getRank() != 2 || !memrefTy.isDynamicDim(0) ||
      !memrefTy.isDynamicDim(1))
    return emitOpError("system memref must be memref<?x?xf64>");
  auto elemTy = dyn_cast<FloatType>(memrefTy.getElementType());
  if (!elemTy || elemTy.getWidth() != 64)
    return emitOpError("system element type must be f64");

  auto rhsTy = dyn_cast<TempType>(getRhs().getType());
  auto resTy = dyn_cast<TempType>(getResult().getType());
  if (!rhsTy || !resTy)
    return emitOpError("rhs/result must be TempType");
  auto rhsElem = dyn_cast<FloatType>(rhsTy.getElementType());
  auto resElem = dyn_cast<FloatType>(resTy.getElementType());
  if (!rhsElem || rhsElem.getWidth() != 64 || !resElem || resElem.getWidth() != 64)
    return emitOpError("rhs/result Temp element type must be f64");
  return success();
}

LogicalResult SolveNonlinearOp::verify() {
  return success();
}

LogicalResult TimeAdvanceOp::verify() {
  // 1) state/result type 必须一致
  auto stTy  = getState().getType();
  auto outTy = getResult().getType();
  if (stTy != outTy)
    return emitOpError("result type must match state type, got state=")
           << stTy << " result=" << outTy;

  // 2) dt 必须是“标量数值”：float / integer / index
  mlir::Type dtTy = getDt().getType();

  // 不允许 shaped（比如 tensor/memref）
  if (isa<mlir::ShapedType>(dtTy))
    return emitOpError("dt must be a scalar numeric type, but got shaped type ")
           << dtTy;

  if (isa<mlir::FloatType>(dtTy)) {
    return mlir::success();
  }
  if (isa<mlir::IndexType>(dtTy)) {
    return mlir::success();
  }
  if (auto it = dyn_cast<mlir::IntegerType>(dtTy)) {
    // i1 作为 dt 基本没意义，直接拒绝；宽度太大也拒绝（lowering 要转到 i64/f64）
    if (it.getWidth() == 1)
      return emitOpError("dt integer type must not be i1");
    if (it.getWidth() > 64)
      return emitOpError("dt integer width > 64 is not supported (got ")
             << it.getWidth() << ")";
    return mlir::success();
  }

  return emitOpError("dt must be float/index/integer scalar, but got ") << dtTy;
}

LogicalResult NonlinearOpDefOp::verify() {
  return success();
}

LogicalResult ApplyNonLinearOp::verify() {
  return success();
}

LogicalResult TimeAdvanceRuntimeOp::verify() {
  // 1) state/result type 必须一致
  auto stTy  = getState().getType();
  auto outTy = getResult().getType();
  if (stTy != outTy)
    return emitOpError("result type must match state type, got state=")
           << stTy << " result=" << outTy;

  // 2) dt 必须是“标量数值”：float / integer / index
  mlir::Type dtTy = getDt().getType();

  // 不允许 shaped（比如 tensor/memref）
  if (isa<mlir::ShapedType>(dtTy))
    return emitOpError("dt must be a scalar numeric type, but got shaped type ")
           << dtTy;

  if (isa<mlir::FloatType>(dtTy)) {
    return mlir::success();
  }
  if (isa<mlir::IndexType>(dtTy)) {
    return mlir::success();
  }
  if (auto it = dyn_cast<mlir::IntegerType>(dtTy)) {
    // i1 作为 dt 基本没意义，直接拒绝；宽度太大也拒绝（lowering 要转到 i64/f64）
    if (it.getWidth() == 1)
      return emitOpError("dt integer type must not be i1");
    if (it.getWidth() > 64)
      return emitOpError("dt integer width > 64 is not supported (got ")
             << it.getWidth() << ")";
    return mlir::success();
  }

  return emitOpError("dt must be float/index/integer scalar, but got ") << dtTy;
}

void StoreOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // 最保守：写目标 field（按 pointer 语义看待）
  OpOperand &fieldOperand = getOperation()->getOpOperand(1);
  effects.emplace_back(MemoryEffects::Write::get(), &fieldOperand);
}

void AssembleMatrixOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

void SolveLinearOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // TODO: 精细标注 solver 的读写/分配；此处保守为未知副作用
  effects.emplace_back(MemoryEffects::Write::get());
}

void SolveNonlinearOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // TODO: 精细标注求解过程的读写/分配；此处保守为未知副作用
  effects.emplace_back(MemoryEffects::Write::get());
}

void TimeAdvanceOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Time advancement may mutate state; conservatively mark as write.
  effects.emplace_back(MemoryEffects::Write::get());
}

void TimeAdvanceRuntimeOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Time advancement may mutate state; conservatively mark as write.
  effects.emplace_back(MemoryEffects::Write::get());
}