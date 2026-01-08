set(LLVM_OPTIONAL_SOURCES bindings/NeptuneModule.cpp)

pybind11_add_module(_neptune_mlir bindings/NeptuneModule.cpp)
target_link_libraries(_neptune_mlir PRIVATE
  NeptuneCompiler
  MLIRNeptuneIRTransforms
  MLIRTransforms
  MLIRIR 
  MLIRPass
  LLVMSupport
  LLVMX86CodeGen
  LLVMX86Info
  LLVMX86Desc
  LLVMAsmParser
  LLVMMC
  LLVMTarget
  LLVMCore
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRMathDialect
  MLIRControlFlowDialect
  MLIRNeptuneIR
  MLIRArithToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRSCFToControlFlow
  MLIRMathToLLVM
  MLIRControlFlowToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRReconcileUnrealizedCasts
)

set_target_properties(_neptune_mlir PROPERTIES 
    INTERPROCEDURAL_OPTIMIZATION OFF
    CXX_VISIBILITY_PRESET "default"
)

add_custom_command(TARGET _neptune_mlir POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_neptune_mlir>
    ${CMAKE_CURRENT_SOURCE_DIR}/neptune/
)
