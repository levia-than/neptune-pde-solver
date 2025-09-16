add_mlir_library(MLIRNeptuneImpl
    NptImplDialect.cpp
    NptImplOps.cpp
    
    DEPENDS
    MLIRNptImplOpsIncGen
)