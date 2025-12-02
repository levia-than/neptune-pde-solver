/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-11-14 11:42:00
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-14 11:46:37
 * @FilePath: /neptune-pde-solver/include/Frontend/NeptuneModuleBuilder.h
 * @Description: 
 * 
 * Copyright (c) 2025 by leviathan, All Rights Reserved. 
 */
#ifndef FRONTEND_NEPTUNEMODULEBUILDER_H
#define FRONTEND_NEPTUNEMODULEBUILDER_H

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include <string>
#include <vector>

namespace mlir::Neptune::Frontend {

/// 极简 1D 线性 stencil 描述，供 Python/pybind 转译层使用。
struct LinearStencil1DDesc {
  int64_t radius = 0;               ///< 支持的邻域半径
  std::vector<double> coeffs;       ///< 长度应为 2*radius+1
  double scale = 1.0;               ///< 额外缩放
};

/// 目前只支持线性 stencil 的一维算子列表。
struct StencilProgram1DDesc {
  int64_t nx = 0;                   ///< 全局网格大小（含边界）
  Type elementType;                 ///< 元素标量类型（通常是 f64/f32）
  std::vector<LinearStencil1DDesc> ops;
};

/// 从 StencilProgram1DDesc 构造 NeptuneIR Module 的小工具。
/// 目标：让上层只关心 “怎么落到 Dialect”，后端只管优化 pass。
class NeptuneModuleBuilder {
public:
  explicit NeptuneModuleBuilder(MLIRContext &ctx);

  /// 根据描述生成一个带单个 `func.func @stencil_step` 的 Module。
  /// 函数签名：(%u_in: memref<nxf>, %u_out: memref<nxf>, %dt: f)
  /// 输出 MLIR ModuleOp，调用方可继续跑 pass pipeline。
  ModuleOp build1DStencilModule(const StencilProgram1DDesc &program,
                                StringRef funcName = "stencil_step");

  /// 便捷函数：直接返回文本 MLIR（ModuleOp 打印）。
  std::string buildModuleString(const StencilProgram1DDesc &program,
                                StringRef funcName = "stencil_step");

private:
  MLIRContext &ctx;
};

} // namespace mlir::Neptune::Frontend

#endif
