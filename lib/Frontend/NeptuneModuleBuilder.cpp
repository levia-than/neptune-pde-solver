/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-11-14 11:42:00
 * @LastEditors: Codex
 * @LastEditTime: 2025-11-30 11:45:00
 * @FilePath: /neptune-pde-solver/lib/Frontend/NeptuneModuleBuilder.cpp
 * @Description:
 *   通过一个极简描述结构，把 1D stencil program 降成 NeptuneIR Module。
 *   预期由 Python/pybind 调用，上层只关心 Dialect 级别的 IR 生成。
 */

#include "Frontend/NeptuneModuleBuilder.h"

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;
using namespace mlir::Neptune::Frontend;

namespace {

static DenseI64ArrayAttr makeDenseI64(MLIRContext *ctx,
                                      ArrayRef<int64_t> values) {
  return DenseI64ArrayAttr::get(ctx, values);
}

static BoundsAttr makeBounds(MLIRContext *ctx, ArrayRef<int64_t> lb,
                             ArrayRef<int64_t> ub) {
  return BoundsAttr::get(ctx, makeDenseI64(ctx, lb), makeDenseI64(ctx, ub));
}

static Value createFloatConst(OpBuilder &b, Location loc, Type elemTy,
                              double v) {
  auto floatTy = dyn_cast<FloatType>(elemTy);
  if (!floatTy)
    return {};
  auto attr = b.getFloatAttr(floatTy, v);
  return b.create<arith::ConstantOp>(loc, elemTy, attr);
}

} // namespace

NeptuneModuleBuilder::NeptuneModuleBuilder(MLIRContext &ctx) : ctx(ctx) {
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
                  NeptuneIRDialect>();
}

ModuleOp NeptuneModuleBuilder::build1DStencilModule(
    const StencilProgram1DDesc &program, StringRef funcName) {
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();

  // ---- 校验输入描述 ----
  if (program.nx <= 0)
    llvm::report_fatal_error("StencilProgram1DDesc.nx must be > 0");
  if (program.ops.empty())
    llvm::report_fatal_error("StencilProgram1DDesc.ops must not be empty");

  Type elemTy = program.elementType;
  if (!elemTy)
    elemTy = b.getF64Type();

  int64_t maxRadius = 0;
  for (const auto &op : program.ops) {
    maxRadius = std::max(maxRadius, op.radius);
    if (static_cast<int64_t>(op.coeffs.size()) != 2 * op.radius + 1)
      llvm::report_fatal_error("coeffs length must equal 2*radius+1");
  }
  if (program.nx <= 2 * maxRadius)
    llvm::report_fatal_error("nx is too small for the stencil radius");

  // ---- 预先准备各种 Attr/Type ----
  SmallVector<int64_t, 1> lbFull{0}, ubFull{program.nx};
  SmallVector<int64_t, 1> lbInterior{maxRadius}, ubInterior{program.nx - maxRadius};

  BoundsAttr boundsFull = makeBounds(&ctx, lbFull, ubFull);
  BoundsAttr boundsInterior = makeBounds(&ctx, lbInterior, ubInterior);
  auto locAttr = NeptuneIR::LocationAttr::get(&ctx, "cell");

  auto fieldTy = FieldType::get(&ctx, elemTy, boundsFull, locAttr);
  auto tempFullTy = TempType::get(&ctx, elemTy, boundsFull, locAttr);
  auto tempInteriorTy = TempType::get(&ctx, elemTy, boundsInterior, locAttr);

  // memref<nx x elemTy>
  auto memrefTy = MemRefType::get({program.nx}, elemTy);

  // ---- 搭一个最简单的 Module + Func ----
  ModuleOp module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(module.getBody());

  FunctionType funcTy =
      b.getFunctionType({memrefTy, memrefTy, elemTy}, {});
  func::FuncOp func = b.create<func::FuncOp>(loc, funcName, funcTy);

  // 入口块
  Block *entry = func.addEntryBlock();
  Value uOld = entry->getArgument(0);
  Value uNew = entry->getArgument(1);
  Value dt = entry->getArgument(2);

  b.setInsertionPointToStart(entry);
  Value fieldOld = b.create<WrapOp>(loc, fieldTy, uOld);
  Value fieldNew = b.create<WrapOp>(loc, fieldTy, uNew);
  Value tmpOld = b.create<LoadOp>(loc, tempFullTy, fieldOld);

  // ---- apply 区域 ----
  auto apply = b.create<ApplyOp>(loc, tempInteriorTy, ValueRange{tmpOld},
                                 boundsInterior, /*shape*/ nullptr);

  // 建 region / block（1D -> 1 个 index 块参）
  Block *body = new Block();
  body->addArgument(b.getIndexType(), loc);
  apply.getBody().push_back(body);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  auto makeAccess = [&](int64_t off) -> Value {
    DenseI64ArrayAttr offsets = makeDenseI64(&ctx, {off});
    return bodyBuilder.create<AccessOp>(loc, elemTy, tmpOld, offsets);
  };

  Value zero = createFloatConst(bodyBuilder, loc, elemTy, 0.0);
  Value rhs; // 延迟初始化，避免多余 add

  for (const auto &opDesc : program.ops) {
    Value acc;
    int64_t r = opDesc.radius;
    for (int64_t k = -r; k <= r; ++k) {
      double coeff = opDesc.coeffs[k + r];
      if (coeff == 0.0)
        continue;
      Value v = makeAccess(k);
      Value c = createFloatConst(bodyBuilder, loc, elemTy, coeff);
      Value term = bodyBuilder.create<arith::MulFOp>(loc, v, c);
      acc = acc ? bodyBuilder.create<arith::AddFOp>(loc, acc, term) : term;
    }
    if (opDesc.scale != 1.0) {
      Value s = createFloatConst(bodyBuilder, loc, elemTy, opDesc.scale);
      acc = bodyBuilder.create<arith::MulFOp>(loc, acc, s);
    }
    rhs = rhs ? bodyBuilder.create<arith::AddFOp>(loc, rhs, acc) : acc;
  }

  if (!rhs)
    rhs = zero;

  // u_new = u_old + dt * rhs（显式 Euler）
  Value center = makeAccess(0);
  Value delta = bodyBuilder.create<arith::MulFOp>(loc, rhs, dt);
  Value updated = bodyBuilder.create<arith::AddFOp>(loc, center, delta);
  bodyBuilder.create<YieldOp>(loc, updated);

  // ---- store 回输出 field（仅内区间）----
  b.setInsertionPointAfter(apply);
  b.create<StoreOp>(loc, apply.getResult(), fieldNew, boundsInterior);
  b.create<func::ReturnOp>(loc);

  return module;
}

std::string NeptuneModuleBuilder::buildModuleString(
    const StencilProgram1DDesc &program, StringRef funcName) {
  ModuleOp module = build1DStencilModule(program, funcName);
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  return text;
}
