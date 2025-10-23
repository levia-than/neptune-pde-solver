#ifndef NEPTUNEIR_NEPTUNEIROPS_H
#define NEPTUNEIR_NEPTUNEIROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROps.h.inc"

#endif