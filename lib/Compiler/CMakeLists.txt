add_mlir_library(NeptuneCompiler
    NeptuneCompiler.cpp
)

target_link_libraries(NeptuneCompiler PUBLIC
    pybind11::pybind11
    MLIROptLib
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRCAPIRegisterEverything
    MLIRNeptuneIR
    MLIRNeptuneIRTransforms
    MLIRNeptuneIRPassesPipeline)

set_source_files_properties(
    NeptuneCompiler.cpp
    PROPERTIES COMPILE_OPTIONS "-fexceptions;-frtti"
)