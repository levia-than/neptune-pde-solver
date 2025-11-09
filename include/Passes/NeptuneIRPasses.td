#ifndef NEPTUNEIR_PASSES_TD
#define NEPTUNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerEvaluateToRealCompute : Pass<"lower-evaluate-to-real-compute"> {
    let summary = "lower evaluate ops to real compute";
    let dependentDialects = [
        "mlir::scf::SCFDialect", 
        "mlir::memref::MemRefDialect", 
        "mlir::arith::ArithDialect", 
        "mlir::func::FuncDialect"
    ];
}

def SymbolicSimplify : Pass<"symbolic-simplify"> {
    let summary = "simplify expressions symbolically";
    let dependentDialects = [
        "mlir::scf::SCFDialect", 
        "mlir::memref::MemRefDialect", 
        "mlir::arith::ArithDialect", 
        "mlir::func::FuncDialect"
    ];
}

#endif
