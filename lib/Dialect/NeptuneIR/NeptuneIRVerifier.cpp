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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

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
  // TODO: 校验 region 的 index 维度 / bounds 维度一致性等
  return success();
}

LogicalResult StoreOp::verify() {
  // TODO: 检查 value/temp 与目标 field elementType & bounds 兼容
  return success();
}

void StoreOp::getEffects(
  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // 最保守：写目标 field（按 pointer 语义看待）
  OpOperand &fieldOperand = getOperation()->getOpOperand(1);
  effects.emplace_back(MemoryEffects::Write::get(), &fieldOperand);
}