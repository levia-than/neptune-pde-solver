// src/Dialect/MyDialect.td
include "mlir/IR/DialectBase.td"

// 1. 定义Dialect
def MyDialect : Dialect {
  let name = "mydialect";          // Dialect名称（需唯一）
  let namespace = "my";            // 操作前缀（如my.foo）
  let cppNamespace = "MyMLIR";     // C++命名空间
  let description = "My custom MLIR dialect for testing.";
}

// 2. 引入操作和类型定义
include "MyOps.td"
include "MyTypes.td"
