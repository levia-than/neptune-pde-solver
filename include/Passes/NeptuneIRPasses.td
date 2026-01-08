#ifndef NEPTUNEIR_PASSES_TD
#define NEPTUNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def NeptuneIRVerifyAnnotatePass : Pass<"neptune-ir-verify-annotate", "mlir::ModuleOp"> {
  let summary = "verify NeptuneIR contracts and attach structure keys";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

def NeptuneIRHighLevelConvertionPass : Pass<"neptune-ir-high-level-convertion", "mlir::ModuleOp"> {
  let summary = "convert high level op, such as TimeAdvanceOp to linear-related op";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

def NeptuneIRStructureLoweringPass : Pass<"neptune-ir-structure-lowering", "mlir::ModuleOp"> {
  let summary = "lower NeptuneIR structure: opdef/apply_{linear,nonlinear} to func";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

def NeptuneIRDataflowLoweringPass : Pass<"neptune-ir-dataflow-lowering", "mlir::ModuleOp"> {
  let summary = "lower NeptuneIR dataflow: apply/access/storage/reduce";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::gpu::GPUDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];

  let options = [
    Option<"backend", "backend",
           "::mlir::Neptune::NeptuneIR::DataflowBackend",
           /*default=*/"::mlir::Neptune::NeptuneIR::DataflowBackend::cpu",
           "dataflow backend", [{
             llvm::cl::values(
               clEnumValN(::mlir::Neptune::NeptuneIR::DataflowBackend::cpu,
                          "cpu", "Lower apply/access to SCF+memref (CPU)"),
               clEnumValN(::mlir::Neptune::NeptuneIR::DataflowBackend::gpu,
                          "gpu", "Lower apply/access to gpu.launch (GPU)")
             )
           }]>,
  ];
}


def NeptuneIRRuntimeLoweringPass : Pass<"neptune-ir-runtime-lowering", "mlir::ModuleOp"> {
  let summary = "lower NeptuneIR runtime ops to shim calls (petsc/cuda/hip/native)";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];

  let options = [
    Option<"runtime", "runtime",
           "::mlir::Neptune::NeptuneIR::RuntimeKind",
           /*default=*/"::mlir::Neptune::NeptuneIR::RuntimeKind::petsc",
           "runtime kind", [{
             llvm::cl::values(
               clEnumValN(::mlir::Neptune::NeptuneIR::RuntimeKind::petsc,
                          "petsc", "Lower to PETSc shim runtime"),
               clEnumValN(::mlir::Neptune::NeptuneIR::RuntimeKind::cuda,
                          "cuda", "Lower to CUDA libraries/runtime"),
               clEnumValN(::mlir::Neptune::NeptuneIR::RuntimeKind::hip,
                          "hip", "Lower to HIP libraries/runtime"),
               clEnumValN(::mlir::Neptune::NeptuneIR::RuntimeKind::native,
                          "native", "Lower to a built-in/native runtime")
             )
           }]>,
  ];
}


#endif
