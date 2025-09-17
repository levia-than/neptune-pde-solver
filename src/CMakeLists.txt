set(LIBS
    MLIROptLib
    MLIRCAPIRegisterEverything
    MLIRNeptuneImpl
)

add_llvm_executable(neptune-opt neptune.cpp)

llvm_update_compile_flags(neptune-opt)
target_link_libraries(neptune-opt PRIVATE ${LIBS})

install(TARGETS neptune-opt
        RUNTIME DESTINATION bin)
