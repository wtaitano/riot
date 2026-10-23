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

# Unit test executable. New unit-test translation units should be appended to
# the source list below. The custom Catch2 main (catch2_define.cpp) initializes
# and finalizes Kokkos so tests can launch device kernels.
add_executable(
  ckm_unit_tests
  catch2_define.cpp
  test_tt_smoke.cpp
)

target_link_libraries(ckm_unit_tests PRIVATE
  Catch2::Catch2
  ckmlib
)

include(Catch)
catch_discover_tests(ckm_unit_tests PROPERTIES TIMEOUT 60)
