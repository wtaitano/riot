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

set (SRC_LIST
  # alphabetical
  tt_smoke.cpp
  tt_smoke.hpp
)

add_library(ckmlib ${SRC_LIST})
target_include_directories(ckmlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ckmlib PUBLIC cxx_std_20)
target_link_libraries(ckmlib PUBLIC parthenon)

if (Kokkos_ENABLE_CUDA)
  target_compile_options(ckmlib PUBLIC --expt-relaxed-constexpr)
endif()

target_compile_options(ckmlib
    PUBLIC
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wno-psabi>
)

add_executable(ckm main.cpp)
target_link_libraries(ckm PRIVATE ckmlib)

install(TARGETS ckm
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
set_target_properties(ckm PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
