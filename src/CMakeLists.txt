set(LIBS
    MLIROptLib
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRCAPIRegisterEverything
    MLIRNeptuneIR
    MLIRNeptuneIRTransforms
)

add_llvm_executable(neptune-opt neptuneOpt.cpp)

llvm_update_compile_flags(neptune-opt)
target_link_libraries(neptune-opt PRIVATE ${LIBS})

install(TARGETS neptune-opt
        RUNTIME DESTINATION bin)
