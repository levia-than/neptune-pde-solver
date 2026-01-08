if(NEPTUNE_ENABLE_PETSC)
    add_library(NeptunePETScRuntime SHARED
        NeptunePETScRuntime.cpp
    )
    target_link_libraries(NeptunePETScRuntime 
        PUBLIC 
            PETSc::PETSc
        PRIVATE
            ${CMAKE_DL_LIBS})
    set_property(TARGET NeptunePETScRuntime PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()