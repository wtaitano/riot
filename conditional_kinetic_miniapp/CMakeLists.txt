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

# Conditional kinetic miniapp (ckm). Built as part of riot: configure riot with
#   cmake -S . -B build -DRIOT_BUILD_CKM=ON
# ckm uses riot's Parthenon (external/parthenon, tensor-train branch), Kokkos, MPI and
# HDF5 settings, and may use riot's src/ (e.g. utilities through riotlib). riot never
# depends on ckm.

if(NOT RIOT_BUILD_PARTHENON)
  message(FATAL_ERROR "ckm needs riot's bundled tensor-train Parthenon "
    "(RIOT_BUILD_PARTHENON=ON).")
endif()
if(NOT RIOT_ENABLE_HDF5)
  message(FATAL_ERROR "ckm requires HDF5; configure riot with RIOT_ENABLE_HDF5=ON.")
endif()

# ckm src
add_subdirectory(src)

if(RIOT_ENABLE_UNIT_TESTS)
  add_subdirectory(tst)
endif()
