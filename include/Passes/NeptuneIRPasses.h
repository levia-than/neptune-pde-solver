//===- LowerEvaluateToLoops.h - pass declaration ---------------*- C++ -*-===//
//
// LowerEvaluateToLoops pass: skeleton that finds evaluate ops and lowers them.
//
//===----------------------------------------------------------------------===//
#ifndef NEPTUNEIR_LOWEREVALUATETOLOOPS_H
#define NEPTUNEIR_LOWEREVALUATETOLOOPS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include <memory>

namespace mlir::Neptune::NeptuneIR {
    
enum class DataflowBackend {
  cpu,
  gpu,
};

enum class RuntimeKind {
  petsc,
  cuda,
  hip,
  native,
};

#define GEN_PASS_DECL
#include "Passes/NeptuneIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes/NeptuneIRPasses.h.inc"

} // namespace mlir::Neptune::NeptuneIR

#endif // NEPTUNEIR_LOWEREVALUATETOLOOPS_H
