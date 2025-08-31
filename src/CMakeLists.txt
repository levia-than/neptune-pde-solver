# src/CMakeLists.txt
# 递归添加子目录（Dialect和Pass）
add_subdirectory(Dialect)
add_subdirectory(Pass)

# 可选：创建一个工具（如自定义mlir-opt）
add_llvm_executable(my-mlir-opt
  my-mlir-opt.cpp

  DEPENDS
  MyMLIRDialect
  MyMLIRPasses
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

# 链接依赖库
target_link_libraries(my-mlir-opt PRIVATE
  MyMLIRDialect
  MyMLIRPasses
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

# 安装工具（可选）
install(TARGETS my-mlir-opt
  RUNTIME DESTINATION bin
)