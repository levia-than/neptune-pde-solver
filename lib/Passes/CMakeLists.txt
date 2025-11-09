add_mlir_library(MLIRNeptuneIRTransforms
    LowerEvaluateToRealCompute.cpp
    SymbolicSimplify.cpp
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