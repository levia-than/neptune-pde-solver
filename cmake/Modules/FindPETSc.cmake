# SPDX-License-Identifier: Apache-2.0
# Minimal FindPETSc module:
# - prefers pkg-config (PETSc/PETSC)
# - supports prefix installs via CMAKE_PREFIX_PATH or PETSC_DIR
# - if PETSC_ARCH is provided, also supports build-tree layout (PETSC_DIR/PETSC_ARCH)

include(FindPackageHandleStandardArgs)

# Collect candidate prefixes (prefer arch/prefix-specific paths first).
set(_petsc_prefixes "")
if(PETSC_DIR)
  if(PETSC_ARCH)
    list(APPEND _petsc_prefixes "${PETSC_DIR}/${PETSC_ARCH}")
  endif()
  list(APPEND _petsc_prefixes "${PETSC_DIR}")
endif()
if(NOT PETSC_DIR AND CMAKE_PREFIX_PATH)
  list(APPEND _petsc_prefixes ${CMAKE_PREFIX_PATH})
endif()
list(REMOVE_DUPLICATES _petsc_prefixes)

# Build PKG_CONFIG_PATH candidates.
set(_petsc_pkg_paths "")
foreach(_p IN LISTS _petsc_prefixes)
  list(APPEND _petsc_pkg_paths
       "${_p}/lib/pkgconfig"
       "${_p}/lib64/pkgconfig")
endforeach()

# Try pkg-config first.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  set(_save_PKG_CONFIG_PATH "$ENV{PKG_CONFIG_PATH}")
  if(_petsc_pkg_paths)
    # Prepend our candidate pkgconfig dirs.
    string(JOIN ":" _joined_paths ${_petsc_pkg_paths} "$ENV{PKG_CONFIG_PATH}")
    set(ENV{PKG_CONFIG_PATH} "${_joined_paths}")
  endif()

  # Try both module names.
  pkg_check_modules(PETSC_PKG QUIET IMPORTED_TARGET PETSc)
  if(NOT PETSC_PKG_FOUND)
    pkg_check_modules(PETSC_PKG QUIET IMPORTED_TARGET PETSC)
  endif()

  set(ENV{PKG_CONFIG_PATH} "${_save_PKG_CONFIG_PATH}")
endif()

if(PETSC_PKG_FOUND)
  set(PETSC_INCLUDE_DIRS ${PETSC_PKG_INCLUDE_DIRS} CACHE STRING "" FORCE)
  # Prefer linking via the imported pkg-config target (handles deps/order best).
  set(PETSC_LIBRARIES PkgConfig::PETSC_PKG CACHE STRING "" FORCE)
  set(PETSC_VERSION ${PETSC_PKG_VERSION} CACHE STRING "" FORCE)
endif()

# Fallback: path-based search (best-effort; may miss static deps).
if(NOT PETSC_PKG_FOUND)
  find_path(PETSC_CONF_INCLUDE_DIR
            NAMES petscconf.h
            HINTS ${_petsc_prefixes}
            PATH_SUFFIXES include)
  find_path(PETSC_INCLUDE_DIR
            NAMES petscsys.h
            HINTS ${_petsc_prefixes}
            PATH_SUFFIXES include)
  find_library(PETSC_LIBRARY
               NAMES petsc
               HINTS ${_petsc_prefixes}
               PATH_SUFFIXES lib lib64)

  set(_petsc_includes ${PETSC_INCLUDE_DIR} ${PETSC_CONF_INCLUDE_DIR})
  list(REMOVE_DUPLICATES _petsc_includes)
  set(PETSC_INCLUDE_DIRS "${_petsc_includes}" CACHE STRING "" FORCE)
  if(PETSC_LIBRARY)
    set(PETSC_LIBRARIES ${PETSC_LIBRARY} CACHE STRING "" FORCE)
  endif()
endif()

# Provide both PETSC_* and PETSc_* variants.
if(PETSC_INCLUDE_DIRS AND NOT PETSc_INCLUDE_DIRS)
  set(PETSc_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})
endif()
if(PETSC_LIBRARIES AND NOT PETSc_LIBRARIES)
  set(PETSc_LIBRARIES ${PETSC_LIBRARIES})
endif()
if(PETSC_VERSION AND NOT PETSc_VERSION)
  set(PETSc_VERSION ${PETSC_VERSION})
endif()

# Derive version from petscversion.h if pkg-config did not supply it.
if(NOT PETSC_VERSION AND PETSC_INCLUDE_DIRS)
  set(_petsc_version_header "")
  foreach(_dir IN LISTS PETSC_INCLUDE_DIRS)
    if(EXISTS "${_dir}/petscversion.h")
      set(_petsc_version_header "${_dir}/petscversion.h")
      break()
    endif()
  endforeach()
  if(_petsc_version_header)
    file(STRINGS "${_petsc_version_header}" _ver_lines
         REGEX "#define PETSC_VERSION_(MAJOR|MINOR|SUBMINOR)")
    foreach(_l IN LISTS _ver_lines)
      if(_l MATCHES "PETSC_VERSION_MAJOR[ \t]+([0-9]+)")
        set(_petsc_ver_major "${CMAKE_MATCH_1}")
      endif()
      if(_l MATCHES "PETSC_VERSION_MINOR[ \t]+([0-9]+)")
        set(_petsc_ver_minor "${CMAKE_MATCH_1}")
      endif()
      if(_l MATCHES "PETSC_VERSION_SUBMINOR[ \t]+([0-9]+)")
        set(_petsc_ver_patch "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    if(_petsc_ver_major AND _petsc_ver_minor)
      if(NOT _petsc_ver_patch)
        set(_petsc_ver_patch 0)
      endif()
      set(PETSC_VERSION "${_petsc_ver_major}.${_petsc_ver_minor}.${_petsc_ver_patch}" CACHE STRING "" FORCE)
    endif()
  endif()
endif()

find_package_handle_standard_args(PETSc
  REQUIRED_VARS PETSC_LIBRARIES PETSC_INCLUDE_DIRS
  VERSION_VAR PETSC_VERSION)

# Create uniform imported target.
if(PETSc_FOUND AND NOT TARGET PETSc::PETSc)
  add_library(PETSc::PETSc INTERFACE IMPORTED)
  target_include_directories(PETSc::PETSc INTERFACE ${PETSC_INCLUDE_DIRS})
  target_link_libraries(PETSc::PETSc INTERFACE ${PETSC_LIBRARIES})
  if(NOT PETSc_FIND_QUIETLY)
    message(STATUS "Found PETSc version: ${PETSC_VERSION}")
    message(STATUS "  Includes: ${PETSC_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${PETSC_LIBRARIES}")
  endif()
endif()

mark_as_advanced(PETSC_INCLUDE_DIR PETSC_LIBRARY)
