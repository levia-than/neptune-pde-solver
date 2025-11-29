add_mlir_library(MLIRNeptuneIRTransforms
    NormalizeNeptuneIRStorage.cpp
    NeptuneIRStencilToSCF.cpp
    DEPENDS MLIRNeptuneIRTransformsIncGen

    LINK_LIBS MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
)