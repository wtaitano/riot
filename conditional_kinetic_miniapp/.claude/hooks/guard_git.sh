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
# Claude Code PreToolUse hook for Bash. Enforces ckm's git rules on agent commands:
#  * no push to lanl/riot (the `upstream` remote or a lanl/riot URL);
#  * no force-push of any kind (--force, -f, --force-with-lease, +refspec);
#  * no rebase / reset --hard / branch -D / push --delete while on, or targeting, the
#    shared branches (wtaitano/ckm, main).
# Rebasing your own topic branch is allowed. Best-effort text matching; GitHub
# rulesets on the fork are the real enforcement.
#
# Input: hook JSON on stdin. Exit 0 = allow, exit 2 = block (stderr goes to Claude).
set -uo pipefail

SHARED_BRANCHES='wtaitano/ckm|main'
# git global options before the subcommand, e.g. `-C dir`, `-c k=v`, `--no-pager`.
GLOBAL_OPTS='( +(-[Cc] +[^ ]+|-[^ ]+))*'

input=$(cat)
cmd=$(printf '%s' "${input}" | python3 -I -c '
import json, sys
print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
' 2> /dev/null) || exit 0
[ -z "${cmd}" ] && exit 0

# Only look at commands that invoke git.
printf '%s' "${cmd}" | grep -qE '(^|[;&|(` ])git( |$)' || exit 0

block() {
  echo "Blocked by ckm guard_git hook: $1" >&2
  echo "ckm never rewrites shared history or pushes to lanl/riot. Ask the human." >&2
  exit 2
}

current_branch() {
  local dir
  dir=$(printf '%s' "${input}" | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("cwd",""))' 2> /dev/null)
  git -C "${dir:-.}" symbolic-ref --short -q HEAD 2> /dev/null || true
}

if printf '%s' "${cmd}" | grep -qE "git${GLOBAL_OPTS} +push"; then
  printf '%s' "${cmd}" | grep -qE 'push( +[^ ]+)* +upstream( |$)|lanl/riot' &&
    block "push to lanl/riot (upstream). PRs into lanl/riot are opened by a human."
  printf '%s' "${cmd}" | grep -qE 'push.*( --force(-with-lease)?(=| |$)| -[a-zA-Z]*f[a-zA-Z]*( |$)| \+[^ ]+)' &&
    block "force-push."
  printf '%s' "${cmd}" | grep -qE "push.*( --delete| -d( |$)| :(${SHARED_BRANCHES})( |$))" &&
    printf '%s' "${cmd}" | grep -qE "(${SHARED_BRANCHES})( |$)" &&
    block "deleting a shared branch."
fi

if printf '%s' "${cmd}" | grep -qE "git${GLOBAL_OPTS} +(rebase|reset +--hard|branch +-D)"; then
  branch=$(current_branch)
  if printf '%s' "${branch}" | grep -qxE "${SHARED_BRANCHES}" ||
    printf '%s' "${cmd}" | grep -qE "(rebase|reset --hard|branch -D)[^;&|]* (origin/|ckm/)?(${SHARED_BRANCHES})( |$)"; then
    block "rebase/reset --hard/branch -D on shared branch (${branch:-target})."
  fi
fi

exit 0
