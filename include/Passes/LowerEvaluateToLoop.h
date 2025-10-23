//===- LowerEvaluateToLoops.h - pass declaration ---------------*- C++ -*-===//
//
// LowerEvaluateToLoops pass: skeleton that finds evaluate ops and lowers them.
//
//===----------------------------------------------------------------------===//
#ifndef NPTIMPL_LOWEREVALUATETOLOOPS_H
#define NPTIMPL_LOWEREVALUATETOLOOPS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include <memory>

namespace mlir {
namespace Neptune {
namespace NeptuneIR {

#define GEN_PASS_DECL
#include "Passes/NeptuneIRPasses.h.inc"

std::unique_ptr<Pass> createLowerEvaluateToLoopPass();

#define GEN_PASS_REGISTRATION
#include "Passes/NeptuneIRPasses.h.inc"

} // namespace NeptuneIR
} // namespace Neptune
} // namespace mlir

#endif // NPTIMPL_LOWEREVALUATETOLOOPS_H
