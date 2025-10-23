#include "Passes/LowerEvaluateToLoop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Dialect/NeptuneIR/NeptuneIRDialect.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"

using namespace mlir;
using namespace mlir::Neptune::NeptuneIR;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, NeptuneIRDialect,
              func::FuncDialect, scf::SCFDialect>();
  mlir::registerAllPasses();
  Neptune::NeptuneIR::registerPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Neptune optimizer driver.\n", registry));
}