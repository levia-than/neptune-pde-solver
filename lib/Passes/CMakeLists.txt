add_mlir_library(MLIRNeptuneIRTransforms
    LowerEvaluateToLoop.cpp
    DEPENDS MLIRNeptuneIRTransformsIncGen
)