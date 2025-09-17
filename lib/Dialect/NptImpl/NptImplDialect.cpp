// lib/Dialect/NptImpl/NptImplDialect.cpp

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"          // 通常会间接包含很多 MLIR 基础
#include "mlir/IR/OpImplementation.h" // 如果你需要 op helper
#include "mlir/Support/TypeID.h" // <--- 明确包含 TypeID 宏定义的位置
#include "Dialect/NptImpl/NptImplDialect.h"
#include "Dialect/NptImpl/NptImplOps.h"


#include "Dialect/NptImpl/NptImplOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/NptImpl/NptImplOps.cpp.inc"

using namespace mlir;
using namespace mlir::Neptune::NptImpl;

void NptImplDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/NptImpl/NptImplOps.cpp.inc"
      >();
}
