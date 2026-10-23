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
# Configure a ckm contributor's riot clone:
#   origin   -> your own fork of lanl/riot (fetch + push)
#   ckm      -> the ckm integration fork (fetch only)
#   upstream -> lanl/riot (fetch only)
# and set pull to never rebase. Run from inside a clone of your fork.
#
# Usage: setup_remotes.sh <your-github-user> [ckm-fork-owner]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <your-github-user> [ckm-fork-owner (default: wtaitano)]" >&2
  exit 2
fi
ME=$1
CKM_OWNER=${2:-wtaitano}

cd "$(git rev-parse --show-toplevel)"

set_remote() {
  local name=$1 url=$2
  if git remote get-url "${name}" > /dev/null 2>&1; then
    git remote set-url "${name}" "${url}"
  else
    git remote add "${name}" "${url}"
  fi
}

set_remote origin "git@github.com:${ME}/riot.git"
if [ "${ME}" != "${CKM_OWNER}" ]; then
  set_remote ckm "https://github.com/${CKM_OWNER}/riot.git"
  git remote set-url --push ckm DISABLED
fi
set_remote upstream "https://github.com/lanl/riot.git"
git remote set-url --push upstream DISABLED

git config push.default current
git config push.autoSetupRemote true
git config pull.rebase false
git config pull.ff only

git remote -v
