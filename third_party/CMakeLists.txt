# third_party/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(ThirdPartyMLIR LANGUAGES CXX C)

# 1. 配置LLVM+MLIR构建选项
set(LLVM_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project)
set(MLIR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir-build)   # 构建目录
set(MLIR_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)   # 安装目录

# 2. 定义MLIR构建目标（作为外部项目）
include(ExternalProject)
ExternalProject_Add(
  MLIRExternal                  # 目标名称
  PREFIX ${MLIR_BUILD_DIR}      # 构建临时文件前缀
  SOURCE_DIR ${LLVM_PROJECT_DIR}# 源码目录（llvm-project根目录）
  BINARY_DIR ${MLIR_BUILD_DIR}  # 构建目录
  # CMake配置选项（关键！）
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL_DIR}  # 安装路径
    -DCMAKE_BUILD_TYPE=Release                  # 构建类型（Release/Debug）
    -DLLVM_ENABLE_PROJECTS=mlir                 # 启用MLIR项目
    -DLLVM_TARGETS_TO_BUILD=host                # 仅构建主机架构（加快速度）
    -DLLVM_ENABLE_ASSERTIONS=ON                 # 启用断言（调试必备）
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF           # 禁用Python绑定（减少依赖）
    -DLLVM_BUILD_EXAMPLES=OFF                   # 禁用LLVM示例（加快构建）
    -DMLIR_BUILD_EXAMPLES=OFF                   # 禁用MLIR示例（加快构建）
    -DCMAKE_C_COMPILER=gcc                      # C编译器（根据系统调整）
    -DCMAKE_CXX_COMPILER=g++                    # C++编译器（根据系统调整）
  # 构建选项：并行编译（-j8，根据CPU核心数调整）
  BUILD_COMMAND make -j8
  # 安装命令
  INSTALL_COMMAND make install
  # 依赖检查：仅在源码或CMakeLists变化时重新构建
  UPDATE_COMMAND ""
)

# 3. 暴露MLIR安装路径给主项目
set(MLIR_INSTALL_DIR ${MLIR_INSTALL_DIR} PARENT_SCOPE)
