#ifndef NEPTUNEIR_PASSES_TD
#define NEPTUNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerEvaluateToLoop : Pass<"lower-evaluate-to-loop"> {
    let summary = "convert toy to arith";
    let constructor = "::mlir::Neptune::NeptuneIR::createLowerEvaluateToLoopPass()";
}

#endif
