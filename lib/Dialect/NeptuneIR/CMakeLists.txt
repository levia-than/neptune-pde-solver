add_mlir_library(MLIRNeptuneIR
    NeptuneIRDialect.cpp
    NeptuneIRVerifier.cpp
    
    DEPENDS
    MLIRNeptuneIROpsIncGen
)