#include "Passes/NeptuneIRPassesPipeline.h"

using namespace mlir;

void mlir::Neptune::NeptuneIR::buildNeptuneToLLVMPipeline(
    mlir::OpPassManager &pm) {
  using namespace mlir::Neptune::NeptuneIR;

  // 0) NeptuneIR front/mid lowering (Module-level)
  pm.addPass(createNeptuneIRVerifyAnnotatePass());
  pm.addPass(createNeptuneIRHighLevelConvertionPass());
  pm.addPass(createNeptuneIRStructureLoweringPass());

  // Runtime (choose runtime)
  {
    NeptuneIRRuntimeLoweringPassOptions opt;
    opt.runtime = RuntimeKind::petsc;     // or cuda/hip/native
    pm.addPass(createNeptuneIRRuntimeLoweringPass(opt));
  }

  // Dataflow (choose backend)
  {
    NeptuneIRDataflowLoweringPassOptions opt;
    opt.backend = DataflowBackend::cpu;   // or DataflowBackend::gpu
    pm.addPass(createNeptuneIRDataflowLoweringPass(opt));
  }

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());

  // Lower memref ops (e.g. subview) before finalizing descriptors.
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createReconcileUnrealizedCastsPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

void mlir::Neptune::NeptuneIR::registerNeptunePipelines() {
  PassPipelineRegistration<>(
      "neptuneir-to-llvm", "Run passes to lower the NeptuneIR to LLVM IR.",
      mlir::Neptune::NeptuneIR::buildNeptuneToLLVMPipeline);
}