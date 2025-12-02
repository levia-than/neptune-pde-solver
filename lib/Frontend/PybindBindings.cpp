/*
 * Pybind11 bindings for the NeptuneModuleBuilder.
 *  - 输入：Python 侧的 StencilProgram1D（来自 nptdsl）
 *  - 输出：NeptuneIR Module 的文本
 *
 * 默认不启用，需设置 NEPTUNE_ENABLE_PYBIND=ON 且找到 pybind11。
 */

#ifdef NEPTUNE_ENABLE_PYBIND

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Frontend/NeptuneModuleBuilder.h"
#include "mlir/IR/MLIRContext.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::Neptune::Frontend;

namespace {

static StencilProgram1DDesc convertProgram(py::handle program,
                                           MLIRContext &ctx) {
  StencilProgram1DDesc desc;
  py::object field = program.attr("field");
  py::object grid = field.attr("grid");
  desc.nx = grid.attr("nx").cast<int64_t>();

  // TODO: 从 numpy dtype 推断 elementType，目前默认 f64。
  desc.elementType = FloatType::getF64(&ctx);

  py::list ops = program.attr("ops");
  for (py::handle o : ops) {
    py::object linear = o.attr("linear");
    LinearStencil1DDesc opDesc;
    opDesc.radius = linear.attr("radius").cast<int64_t>();
    opDesc.coeffs = py::cast<std::vector<double>>(linear.attr("coeffs"));
    opDesc.scale = linear.attr("scale").cast<double>();
    desc.ops.push_back(std::move(opDesc));
  }
  return desc;
}

} // namespace

PYBIND11_MODULE(_neptune_mlir, m) {
  m.doc() = "Neptune MLIR frontend bindings";
  m.def(
      "build_stencil_module",
      [](py::object program, std::string funcName) {
        MLIRContext ctx;
        NeptuneModuleBuilder builder(ctx);
        auto desc = convertProgram(program, ctx);
        return builder.buildModuleString(desc, funcName);
      },
      py::arg("program"), py::arg("func_name") = "stencil_step");
}

#endif // NEPTUNE_ENABLE_PYBIND
