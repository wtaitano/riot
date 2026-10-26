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
# Enforce the ckm isolation boundary on tracked files. The boundary is one-way:
#  * ckm MAY use riot (src/ utilities, riotlib, riot's Parthenon and Catch2).
#  * riot must NOT depend on ckm. Files outside conditional_kinetic_miniapp/ must not
#    reference the miniapp, except ALLOWED_ROOT_FILES: the root files git/GitHub/CMake
#    need in order to wire ckm in.
#  * ckm must not carry its own copies of riot's dependencies (no submodules under
#    conditional_kinetic_miniapp/); it uses riot's external/.
# Reusing riot *physics* packages is a review guideline (AGENTS.md), not checked here.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
cd "${REPO}"

CKM=conditional_kinetic_miniapp
ALLOWED_ROOT_FILES=(
  .gitmodules
  CMakeLists.txt
  .github/CODEOWNERS
  .github/workflows/ckm.yml
)

status=0

# 1. riot -> ckm. Any tracked file outside the miniapp naming it.
OUTSIDE=$(git ls-files | grep -vE "^${CKM}/" || true)
hits=$(echo "${OUTSIDE}" | xargs grep -lI "${CKM}" 2> /dev/null || true)
for f in ${hits}; do
  allowed=0
  for a in "${ALLOWED_ROOT_FILES[@]}"; do
    [ "${f}" = "${a}" ] && allowed=1
  done
  if [ ${allowed} -eq 0 ]; then
    echo "ERROR: ${f} references ${CKM} (riot must not depend on ckm)."
    echo "       Only these root files may: ${ALLOWED_ROOT_FILES[*]}"
    status=1
  fi
done

# 2. riot's CMake may only add ckm behind RIOT_BUILD_CKM (off by default).
if ! grep -qE 'option\(RIOT_BUILD_CKM .* OFF\)' CMakeLists.txt; then
  if grep -q "${CKM}" CMakeLists.txt; then
    echo "ERROR: CMakeLists.txt references ${CKM} without an off-by-default RIOT_BUILD_CKM option."
    status=1
  fi
fi

# 3. No ckm-owned submodules.
sub=$(git ls-files -s -- "${CKM}" | awk '$1 == "160000" {print $4}')
if [ -n "${sub}" ]; then
  echo "ERROR: ckm must use riot's external/ dependencies, not its own submodules:"
  echo "${sub}"
  status=1
fi

if [ ${status} -eq 0 ]; then
  echo "ckm isolation boundary: OK"
fi
exit ${status}
