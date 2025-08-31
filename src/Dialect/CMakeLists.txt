# src/Dialect/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS MyDialect.td)  # 主TableGen文件

# 1. 生成Dialect头文件（MyDialect.h.inc）
mlir_tablegen(MyDialect.h.inc -gen-dialect-decls)
# 2. 生成Dialect源文件（MyDialect.cpp.inc）
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs)
# 3. 生成操作头文件（MyOps.h.inc）
mlir_tablegen(MyOps.h.inc -gen-op-decls)
# 4. 生成操作源文件（MyOps.cpp.inc）
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
# 5. 生成类型头文件（MyTypes.h.inc）
mlir_tablegen(MyTypes.h.inc -gen-type-decls)
# 6. 生成类型源文件（MyTypes.cpp.inc）
mlir_tablegen(MyTypes.cpp.inc -gen-type-defs)

# 7. 定义TableGen目标（确保编译顺序：先生成代码，再编译源码）
add_public_tablegen_target(MyDialectTableGen)

# 8. 编译自定义Dialect库
add_mlir_dialect_library(MyMLIRDialect
  # 自定义源码
  MyDialect.cpp
  MyOps.cpp
  MyTypes.cpp

  # 依赖TableGen生成的文件（需先执行TableGen目标）
  DEPENDS
  MyDialectTableGen

  # 链接MLIR核心库
  LINK_LIBS PUBLIC
  MLIRIR                # MLIR IR核心
  MLIRSupport           # MLIR工具支持
  MLIRInferTypeOpInterface  # 类型推断接口（可选）
)

# 9. 安装Dialect库（可选）
install(TARGETS MyMLIRDialect
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
