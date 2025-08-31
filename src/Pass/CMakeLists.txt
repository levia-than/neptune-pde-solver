# src/Pass/CMakeLists.txt
add_mlir_library(MyMLIRPasses
  # 自定义Pass源码
  MyPass.cpp

  # 依赖自定义Dialect（确保Dialect先编译）
  DEPENDS
  MyMLIRDialect

  # 链接依赖库
  LINK_LIBS PUBLIC
  MyMLIRDialect  # 自定义Dialect
  MLIRIR         # MLIR IR核心
  MLIRPass       # MLIR Pass框架
  MLIRTransforms # MLIR变换工具
)

# 安装Pass库（可选）
install(TARGETS MyMLIRPasses
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
