add_mlir_library(MLIRNeptuneIR
    NeptuneIRDialect.cpp
    
    DEPENDS
    MLIRNeptuneIROpsIncGen
)