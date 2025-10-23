set(LLVM_TARGET_DEFINITIONS NeptuneIRPasses.td)
mlir_tablegen(NeptuneIRPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRNeptuneIRTransformsIncGen)
