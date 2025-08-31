// src/Dialect/MyDialect.cpp
#include "MyMLIR/Dialect/MyDialect.h"
#include "mlir/IR/DialectImplementation.h"

// 引入TableGen生成的代码（必须在实现前）
#include "MyDialect.cpp.inc"
#include "MyOps.cpp.inc"
#include "MyTypes.cpp.inc"

using namespace mlir;
using namespace MyMLIR;

// 1. 注册Dialect
void MyDialect::initialize() {
  // 注册操作
  addOperations<
#define GET_OP_LIST
#include "MyOps.cpp.inc"
    >();
  // 注册类型（若有）
  addTypes<
#define GET_TYPE_LIST
#include "MyTypes.cpp.inc"
    >();
}

// 2. 定义Dialect的ID（全局唯一）
#include "MyDialect.h.inc"
