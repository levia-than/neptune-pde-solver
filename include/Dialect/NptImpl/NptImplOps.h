#ifndef NPTIMPL_NPTIMPL_OPS_H
#define NPTIMPL_NPTIMPL_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/NptImpl/NptImplOps.h.inc"

#endif