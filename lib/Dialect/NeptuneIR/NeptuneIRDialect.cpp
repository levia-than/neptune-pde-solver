//===- NeptuneIROps.cpp - NeptuneIR ops verifier implementations -------===//
//
// Verifier implementations for NeptuneIR ops.
//
// Replace or add this file in your NeptuneIR library. It intentionally
// minimizes reliance on ODS-generated accessor names and uses generic
// MLIR APIs so it is robust across MLIR versions / small ODS differences.
//
//===----------------------------------------------------------------------===//

#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "Dialect/NeptuneIR/NeptuneIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/TypeSwitch.h" 

// Types implementation (TypeIDResolver etc.)
#define GET_TYPEDEF_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.cpp.inc"

// Ops implementation (print/parse/verify/getEffects, bytecode support ...)
#define GET_OP_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROps.cpp.inc"

// Dialect-level implementation (parse/print helpers, dialect ctor if generated)
#include "Dialect/NeptuneIR/NeptuneIROpsDialect.cpp.inc"


// Helper: check that an Attribute is an ArrayAttr of IntegerAttr.
// static bool isArrayOfIntegerAttr(Attribute attr) {
//   if (!attr) return false;


//   if (auto arr = dyn_cast<ArrayAttr>(attr)) {
//     for (auto a : arr.getValue()) {
//       if (!isa<IntegerAttr>(a))
//         return false;
//     }
//     return true;
//   }
//   return false;
// }

// Helper: simple check whether a Type looks like a memref-like storage.
// Be conservative: recognize MemRefType explicitly; other types (e.g. DescriptorType)
// are accepted by caller if desired.
// static bool isMemRefLikeType(Type t) {
//   return isa<MemRefType>(t);
// }

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

void NeptuneIRDialect::initialize() {
  // Register types (this uses the GET_TYPEDEF_LIST part of the generated file)
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.cpp.inc"
      >();

  // Register ops
  addOperations<
#define GET_OP_LIST
#include "Dialect/NeptuneIR/NeptuneIROps.cpp.inc"
      >();
}