#ifndef NEPTUNEIR_PASSES_TD
#define NEPTUNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def NormalizeNeptuneIRStorage : Pass<"normalize-neptune-ir-storage"> {
    let summary = "normalize neptune ir storage";
    let dependentDialects = [
        "mlir::scf::SCFDialect", 
        "mlir::memref::MemRefDialect", 
        "mlir::arith::ArithDialect", 
        "mlir::func::FuncDialect",
        "Neptune::NeptuneIR::NeptuneIRDialect"
    ];
}

def NeptuneIRStencilToSCF : Pass<"neptune-ir-stencil-to-scf"> {
    let summary = "neptune ir stencil to scf";
    let dependentDialects = [
        "mlir::scf::SCFDialect",
        "mlir::memref::MemRefDialect",
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "Neptune::NeptuneIR::NeptuneIRDialect"
    ];
}

#endif
