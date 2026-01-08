add_mlir_library(MLIRNeptuneIRTransforms
    VerifyAndAnnotate.cpp
    StructureLowering.cpp
    RuntimeLowering.cpp
    DataflowLowering.cpp
    HighLevelConvertion.cpp
    DEPENDS MLIRNeptuneIRTransformsIncGen

    LINK_LIBS MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
)