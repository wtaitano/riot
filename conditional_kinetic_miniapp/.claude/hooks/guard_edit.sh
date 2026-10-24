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
# Claude Code PreToolUse hook for file-editing tools. Blocks edits to files outside
# conditional_kinetic_miniapp/ except the root files git/GitHub require at the top level
# (keep in sync with ALLOWED_ROOT_FILES in script/check_boundary.sh). Paths outside the
# riot checkout (scratch dirs, $TMPDIR, ...) are not this hook's concern and pass.
#
# Input: hook JSON on stdin. Exit 0 = allow, exit 2 = block (stderr goes to Claude).
set -uo pipefail

CKM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
REPO=$(cd "${CKM_DIR}/.." && pwd -P)
ALLOWED_ROOT_FILES=(
  .gitmodules
  .github/CODEOWNERS
  .github/workflows/ckm.yml
)

input=$(cat)
file=$(printf '%s' "${input}" | python3 -I -c '
import json, sys
d = json.load(sys.stdin)
ti = d.get("tool_input", {})
print(ti.get("file_path") or ti.get("notebook_path") or "")
' 2> /dev/null) || exit 0
[ -z "${file}" ] && exit 0

# Resolve to an absolute, normalized path without requiring it to exist.
case "${file}" in
  /*) abs=${file} ;;
  *) abs=$(pwd -P)/${file} ;;
esac
abs=$(python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${abs}")

case "${abs}" in
  "${CKM_DIR}" | "${CKM_DIR}"/*) exit 0 ;;
  "${REPO}"/*) ;;
  *) exit 0 ;;
esac

rel=${abs#"${REPO}"/}
for a in "${ALLOWED_ROOT_FILES[@]}"; do
  [ "${rel}" = "${a}" ] && exit 0
done

cat >&2 << EOF
Blocked by ckm guard_edit hook: ${rel} is outside conditional_kinetic_miniapp/.
ckm must not modify riot files. The only root files ckm may edit are:
  ${ALLOWED_ROOT_FILES[*]}
If this change is truly needed, stop and ask the human; it needs riot-developer approval.
EOF
exit 2
