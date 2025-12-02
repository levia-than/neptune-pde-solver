set(LLVM_OPTIONAL_SOURCES PybindBindings.cpp)

add_mlir_library(MLIRNeptuneFrontend
    NeptuneModuleBuilder.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRNeptuneIR
)

# 可选：pybind11 扩展，生成 Python 模块 _neptune_mlir
if(NEPTUNE_ENABLE_PYBIND)
  find_package(pybind11 QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(_neptune_mlir PybindBindings.cpp)
    target_link_libraries(_neptune_mlir PRIVATE
      MLIRNeptuneFrontend
      MLIRIR
      MLIRFuncDialect
      MLIRArithDialect
      MLIRMemRefDialect
      MLIRNeptuneIR
    )
    target_compile_definitions(_neptune_mlir PRIVATE NEPTUNE_ENABLE_PYBIND)
  else()
    message(STATUS "pybind11 not found, skipping _neptune_mlir")
  endif()
endif()
