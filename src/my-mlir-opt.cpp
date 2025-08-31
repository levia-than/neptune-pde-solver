// src/my-mlir-opt.cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "MyMLIR/Dialect/MyDialect.h"
#include "MyMLIR/Pass/MyPass.h"

int main(int argc, char **argv) {
  // 注册所有标准Dialect
  mlir::registerAllDialects();
  
  // 注册自定义Dialect
  mlir::MyMLIR::MyDialect::registerDialect(*mlir::getGlobalContext());
  
  // 注册所有标准Pass
  mlir::registerAllPasses();
  
  // 注册自定义Pass
  mlir::MyMLIR::registerMyPasses();

  // 运行mlir-opt主函数
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MyMLIR Optimizer",
                       [](mlir::DialectRegistry &registry) {
                         // 向注册表添加自定义Dialect
                         registry.insert<mlir::MyMLIR::MyDialect>();
                       }));
}