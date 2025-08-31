# test/MyDialect/CMakeLists.txt
# 1. 获取mlir-opt工具路径（从third_party安装目录）
set(MLIR_OPT ${MLIR_INSTALL_DIR}/bin/mlir-opt)

# 2. 添加测试用例
add_test(
  NAME MyDialectBasicTest
  COMMAND ${MLIR_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/basic.test -verify-diagnostics
)

# 3. 依赖自定义Dialect库（确保库已编译）
set_tests_properties(MyDialectBasicTest PROPERTIES
  DEPENDS MyMLIRDialect
)
