cmake_minimum_required(VERSION 3.20)
project(NeptuneSolver LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# 3. 配置编译选项（与MLIR对齐）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NEPTUNE_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(NEPTUNE_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# 4. 配置头文件路径
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # 自定义头文件
  ${CMAKE_CURRENT_BINARY_DIR}/include  # 要导入诸如inc等文件
  ${LLVM_INCLUDE_DIRS}                 # LLVM头文件
  ${MLIR_INCLUDE_DIRS}                 # MLIR头文件
)

# 5. 链接LLVM/MLIR的CMake模块（生成TableGen代码必备）
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR} ${MLIR_CMAKE_DIR})
include(TableGen)          # MLIR TableGen工具
include(AddLLVM)           # LLVM编译辅助
include(AddMLIR)           # MLIR编译辅助
include(HandleLLVMOptions) # LLVM编译选项

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 6. 添加自定义源码（Dialect/Pass）
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)

# 7. 前端引入，当前为空
# add_subdirectory(frontend)

# 8. 添加测试（可选，需启用BUILD_TESTING）
add_subdirectory(test)
