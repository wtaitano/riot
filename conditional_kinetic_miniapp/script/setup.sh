#!/bin/bash
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
#
# Fetch the ckm submodules. They are marked `update = none` in .gitmodules so a
# plain `git clone --recursive` of riot skips them; this overrides that for the
# ckm paths only. Safe to re-run.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
cd "${REPO}"

CKM_SUBMODULES=(
  conditional_kinetic_miniapp/external/parthenon
  conditional_kinetic_miniapp/external/Catch2
)

for sm in "${CKM_SUBMODULES[@]}"; do
  echo "Updating ${sm}"
  git -c "submodule.${sm}.update=checkout" submodule update --init "${sm}"
done

# Parthenon's own submodules (Kokkos, ...) use the default update mode.
git -C conditional_kinetic_miniapp/external/parthenon submodule update --init --recursive

git submodule status -- "${CKM_SUBMODULES[@]}"
