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
# Claude Code PreToolUse hook for the project-map subagent (a frontmatter hook in
# agents/project-map.md, active only while that agent runs). Lets its file-writing tools
# touch only conditional_kinetic_miniapp/.project-map/, the miniapp .gitignore, and the
# agent's own memory in ~/.claude/agent-memory/project-map/. Fails closed.
#
# Input: hook JSON on stdin. Exit 0 = allow, exit 2 = block (stderr goes to Claude).
set -uo pipefail

CKM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
MEMORY_DIR=$(python3 -I -c '
import os
print(os.path.realpath(os.path.expanduser("~/.claude/agent-memory/project-map")))
')

input=$(cat)
file=$(printf '%s' "${input}" | python3 -I -c '
import json, sys
ti = json.load(sys.stdin).get("tool_input", {})
print(ti.get("file_path") or ti.get("notebook_path") or "")
' 2> /dev/null) || file=""

abs="(no path)"
if [ -n "${file}" ]; then
  case "${file}" in
    /*) abs=${file} ;;
    *) abs=$(pwd -P)/${file} ;;
  esac
  abs=$(python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${abs}")
fi

case "${abs}" in
  "${CKM_DIR}"/.project-map/* | "${CKM_DIR}"/.gitignore | "${MEMORY_DIR}"/*) exit 0 ;;
esac

echo "Blocked by ckm guard_project_map hook: project-map may only write inside" \
  ".project-map/, the miniapp .gitignore, or its own memory (got ${abs})." >&2
exit 2
