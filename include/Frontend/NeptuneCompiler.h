#ifndef NEPTUNE_COMPILER_H
#define NEPTUNE_COMPILER_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Pybind11 头文件
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include <pybind11/functional.h> // 支持 python function <-> std::function
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 支持 std::vector <-> python list
// MLIR 头文件
#include "mlir/Dialect/Arith/IR/Arith.h"  // 算术运算
#include "mlir/Dialect/Func/IR/FuncOps.h" // 函数
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace py = pybind11;

// 我们把这个传给 Python，Python 再把它传回来。它就是个不透明的句柄。
struct PyValue {
  mlir::Value value;

  PyValue();
  PyValue(mlir::Value v);

  // 方便调试打印
  std::string repr();
};

class NeptuneCompiler {
public:
  NeptuneCompiler();

  // ------------------------------------------------------------------------
  // 基础 Op 构建
  // ------------------------------------------------------------------------

  PyValue createWrap(PyValue buffer, std::string type_hint);

  PyValue createAccess(PyValue temp, std::vector<int64_t> offsets);

  // 算术运算 (Arith Dialect)
  PyValue createArithAdd(PyValue lhs, PyValue rhs);

  PyValue createArithSub(PyValue lhs, PyValue rhs);

  PyValue createArithMul(PyValue lhs, PyValue rhs);

  PyValue createConstant(double value);

  PyValue createApply(std::vector<PyValue> inputs, std::vector<int64_t> lb,
                      std::vector<int64_t> ub, py::function body_builder);

  void createLinearOpDef(std::string name, std::vector<int64_t> lb,
                         std::vector<int64_t> ub, std::string loc_kind,
                         py::function body_builder);

  PyValue createAssembleMatrix(std::string op_symbol);

  PyValue createSolveLinear(PyValue matrix, PyValue rhs, std::string solver,
                            double tol);
  // --- 函数编排 (Strategy Support) ---
  // 开始定义一个 func.func
  void startFunction(std::string name, std::vector<PyValue> arg_types_hints);

  // 结束当前函数定义
  void endFunction();

  // 获取当前函数的参数
  PyValue getFunctionArg(int index);

  // 创建 return
  void createReturn(PyValue retVal);
  // AOT
  std::string compileToObjectFile(std::string output_filename);

  // 调试用
  std::string dump();

private:
  void runPipeline();
  mlir::Location loc();

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::func::FuncOp currentFunc;
};

#endif // NEPTUNE_COMPILER_H
