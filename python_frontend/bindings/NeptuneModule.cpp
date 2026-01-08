#include "Frontend/NeptuneCompiler.h"

// 定义 Python 模块名称为 neptune_backend
PYBIND11_MODULE(_neptune_mlir, m) {
  m.doc() = "NeptuneIR MLIR Backend via Pybind11";

  // 绑定 PyValue
  py::class_<PyValue>(m, "Value").def("__repr__", &PyValue::repr);

  // 绑定 Compiler
  py::class_<NeptuneCompiler>(m, "Compiler")
      .def(py::init<>())
      // 基础
      .def("dump", &NeptuneCompiler::dump)
      .def("create_wrap", &NeptuneCompiler::createWrap)
      .def("create_access", &NeptuneCompiler::createAccess)
      // 算术
      .def("create_arith_add", &NeptuneCompiler::createArithAdd)
      .def("create_arith_sub", &NeptuneCompiler::createArithSub)
      .def("create_arith_mul", &NeptuneCompiler::createArithMul)
      .def("create_constant", &NeptuneCompiler::createConstant)
      // 高级 DSL 核心
      .def("create_apply", &NeptuneCompiler::createApply, py::arg("inputs"),
           py::arg("lb"), py::arg("ub"), py::arg("body_builder"))
      .def("create_linear_opdef", &NeptuneCompiler::createLinearOpDef,
           py::arg("name"), py::arg("lb"), py::arg("ub"), py::arg("loc_kind"),
           py::arg("body_builder"))
      // 求解器
      .def("create_assemble_matrix", &NeptuneCompiler::createAssembleMatrix)
      .def("create_solve_linear", &NeptuneCompiler::createSolveLinear)
      .def("start_function", &NeptuneCompiler::startFunction)
      .def("end_function", &NeptuneCompiler::endFunction)
      .def("get_function_arg", &NeptuneCompiler::getFunctionArg)
      .def("create_return", &NeptuneCompiler::createReturn)
      .def("compile_to_object_file", &NeptuneCompiler::compileToObjectFile);
}
