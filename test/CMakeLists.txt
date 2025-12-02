# Ensure FileCheck is available as a CMake target even when using an installed
# LLVM that does not export it.
if(NOT TARGET FileCheck)
  if(NOT LLVM_FILECHECK_EXE)
    find_program(LLVM_FILECHECK_EXE
                 NAMES FileCheck
                 HINTS
                   ${LLVM_TOOLS_BINARY_DIR}
                   ${PROJECT_SOURCE_DIR}/build/llvm-build/bin
                   ${PROJECT_BINARY_DIR}/../llvm-build/bin
                 DOC "Path to the FileCheck executable used by tests")
  endif()

  if(NOT LLVM_FILECHECK_EXE)
    message(FATAL_ERROR "FileCheck executable not found. Build LLVM with utils or set LLVM_FILECHECK_EXE.")
  endif()

  add_executable(FileCheck IMPORTED GLOBAL)
  set_property(TARGET FileCheck PROPERTY IMPORTED_LOCATION ${LLVM_FILECHECK_EXE})

  # Point lit at the tool directory that actually contains FileCheck.
  get_filename_component(_FILECHECK_DIR ${LLVM_FILECHECK_EXE} DIRECTORY)
  if(NOT EXISTS "${LLVM_TOOLS_BINARY_DIR}/FileCheck")
    set(LLVM_TOOLS_BINARY_DIR ${_FILECHECK_DIR} CACHE PATH "Path to LLVM tools" FORCE)
  endif()
endif()

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir_tests)

add_lit_testsuite(check-neptune "Running neptune regression tests..."
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS FileCheck neptune-opt)

set_target_properties(check-neptune PROPERTIES FOLDER "Tests")

add_lit_testsuites(NEPTUNE
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS FileCheck neptune-opt)
