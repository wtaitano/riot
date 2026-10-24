# ========================================================================================
#  (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
# This file was made in part with generative AI.

# Conditional kinetic miniapp (ckm). A standalone Parthenon application that
# lives inside the riot repository but shares no source or build with it. Build
# it from this directory, never from the riot top level:
#   cmake -S conditional_kinetic_miniapp -B build-ckm

include(GNUInstallDirs)

# Load machine specific defaults (same convention as riot). Must precede project().
if (MACHINE_CFG)
  if(EXISTS "${MACHINE_CFG}")
    include(${MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration at ${MACHINE_CFG} not found.")
  endif()
elseif (DEFINED ENV{MACHINE_CFG})
  if(EXISTS "$ENV{MACHINE_CFG}")
    include($ENV{MACHINE_CFG})
  else()
    message(FATAL_ERROR "Given machine configuration from environment variable "
      "MACHINE_CFG at $ENV{MACHINE_CFG} not found.")
  endif()
endif()

cmake_minimum_required(VERSION 3.26)
project(ckm
  LANGUAGES C CXX
  VERSION 0.1.0
)
# The tensor-train Parthenon branch requires C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

include(CTest)
option(CKM_ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(CKM_ENABLE_CUDA "Enable cuda for ckm and all dependencies" OFF)
option(CKM_ENABLE_MPI "Enable MPI for ckm and all dependencies" ON)
option(CKM_ENABLE_OPENMP "Enable OpenMP for ckm and parthenon" OFF)
option(CKM_ENABLE_WARNINGS "Enable compiler warnings" OFF)
option(CKM_BUILD_CATCH2 "Build included version of Catch2" ON)

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
    "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

set(PARTHENON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon")
if(NOT EXISTS "${PARTHENON_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "The ckm Parthenon submodule is not checked out at ${PARTHENON_DIR}. "
    "Run conditional_kinetic_miniapp/script/setup.sh first.")
endif()

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
    "Kokkos aggressive vectorization" FORCE)

# CUDA
if(CKM_ENABLE_CUDA)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "" FORCE)
endif()

# HDF5 is required for ckm (no off switch).
set(HDF5_PREFER_PARALLEL ${CKM_ENABLE_MPI})
find_package(HDF5 COMPONENTS C HL)
if (NOT HDF5_FOUND)
  message(FATAL_ERROR "HDF5 is required for ckm but couldn't be found. "
    "Please install HDF5 or point CMake to it by adding its path to the "
    "CMAKE_PREFIX_PATH environment variable.")
endif()
if (CKM_ENABLE_MPI AND (NOT HDF5_IS_PARALLEL))
  message(FATAL_ERROR "MPI is enabled but only a serial version of HDF5 was found. "
    "Please install a parallel version of HDF5 (or point CMake to it by adding its "
    "path to the CMAKE_PREFIX_PATH environment variable), or disable MPI by "
    "rerunning CMake with -DCKM_ENABLE_MPI=OFF.")
endif()
set(PARTHENON_DISABLE_HDF5 OFF CACHE BOOL "" FORCE)

if(CKM_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  set(PARTHENON_DISABLE_MPI OFF CACHE BOOL "" FORCE)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

if(CKM_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  set(PAR_LOOP_LAYOUT MDRANGE_LOOP CACHE STRING "" FORCE)
else()
  set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "" FORCE)
endif()

# parthenon (the tensor-train branch, pinned in external/parthenon)
message("Configuring Parthenon")
set(PARTHENON_LINT_DEFAULT OFF CACHE BOOL "" FORCE)
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
set(PARTHENON_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
if (CKM_ENABLE_WARNINGS)
  set(ENABLE_COMPILER_WARNINGS ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(external/parthenon parthenon)

# ckm src
message("\nConfiguring src")
add_subdirectory(src)

if(CKM_ENABLE_UNIT_TESTS)
  message("\nConfiguring tests")
  if(CKM_BUILD_CATCH2)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/CMakeLists.txt")
      message(FATAL_ERROR "The ckm Catch2 submodule is not checked out. "
        "Run conditional_kinetic_miniapp/script/setup.sh first, "
        "or configure with -DCKM_BUILD_CATCH2=OFF to use a system Catch2.")
    endif()
    add_subdirectory(external/Catch2 Catch2)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/external/Catch2/extras)
  else()
    find_package(Catch2 3 REQUIRED)
  endif()
  enable_testing()
  add_subdirectory(tst)
endif()
