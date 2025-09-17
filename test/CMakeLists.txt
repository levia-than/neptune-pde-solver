configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py 
                       MAIN_CONFIG 
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir_test)

add_lit_testsuite(check-neptune "Running neptune regression tests..."
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS FileCheck neptune-opt)

set_target_properties(check-neptune PROPERTIES FOLDER "Tests")

add_lit_testsuites(NEPTUNE
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS FileCheck neptune-opt)