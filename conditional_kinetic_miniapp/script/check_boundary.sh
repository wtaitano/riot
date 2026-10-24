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
# Enforce the ckm isolation boundary on tracked files:
#  1. ckm code and build files (C++, CMake) must not reference riot's source,
#     riot's external dependencies, or riot's build targets.
#  2. Files outside conditional_kinetic_miniapp/ must not reference the miniapp,
#     except for the root files that git or GitHub require to live at the top
#     level (ALLOWED_ROOT_FILES).
# Docs and agent instructions may link to riot's policy docs; only code and
# build files are checked in direction 1.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
cd "${REPO}"

CKM=conditional_kinetic_miniapp
ALLOWED_ROOT_FILES=(
  .gitmodules
  .github/CODEOWNERS
  .github/workflows/ckm.yml
)

status=0

# 1. ckm -> riot. Relative escapes out of the miniapp and riot-only names.
CKM_CODE=$(git ls-files -- "${CKM}" |
  grep -E '\.(cpp|hpp|h|cc|cxx|in)$|(^|/)CMakeLists\.txt$|\.cmake$' |
  grep -vE "^${CKM}/external/" || true)
FORBIDDEN='\.\./(\.\./)*(src|external|tst|cmake)/|riotlib|riot_driver|riot_utils|namespace riot\b|riot::|RIOT_[A-Z_]+|PROJECT_SOURCE_DIR\}/\.\.|CMAKE_SOURCE_DIR\}/(src|external)'
if [ -n "${CKM_CODE}" ]; then
  hits=$(echo "${CKM_CODE}" | xargs grep -nE "${FORBIDDEN}" || true)
  if [ -n "${hits}" ]; then
    echo "ERROR: ckm code references riot (isolation boundary, direction ckm -> riot):"
    echo "${hits}"
    status=1
  fi
fi

# 2. riot -> ckm. Any tracked file outside the miniapp naming it.
OUTSIDE=$(git ls-files | grep -vE "^${CKM}/" || true)
hits=$(echo "${OUTSIDE}" | xargs grep -lI "${CKM}" 2>/dev/null || true)
for f in ${hits}; do
  allowed=0
  for a in "${ALLOWED_ROOT_FILES[@]}"; do
    [ "${f}" = "${a}" ] && allowed=1
  done
  if [ ${allowed} -eq 0 ]; then
    echo "ERROR: ${f} references ${CKM} (isolation boundary, direction riot -> ckm)."
    echo "       Only these root files may: ${ALLOWED_ROOT_FILES[*]}"
    status=1
  fi
done

if [ ${status} -eq 0 ]; then
  echo "ckm isolation boundary: OK"
fi
exit ${status}
