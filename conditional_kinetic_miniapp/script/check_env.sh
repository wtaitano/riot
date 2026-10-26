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
# Diagnose why Claude Code's Bash sandbox (bubblewrap) may fail to start on this machine,
# and print the settings that fix it. Typical symptom:
#   bwrap: Can't mkdir /tmp/claude-empty-XXXX: No such file or directory
# on a system where /tmp is a symlink (for example to /ram/tmp) or not writable.
#
# Claude Code keeps its temp files under ${CLAUDE_CODE_TMPDIR:-<OS temp dir>}/claude-<uid>.
# CLAUDE_CODE_TMPDIR is ignored in project settings, so it cannot be committed to the repo;
# each user sets it once, in their shell or in ~/.claude/settings.json.
#
# Run this in the same kind of shell you start `claude` from. Read-only apart from a
# short-lived probe file inside the temp directory. Exit status: 0 = looks fine, 1 = a
# problem was found.
set -uo pipefail

problems=0
note() { printf '  %s\n' "$*"; }
ok() { printf 'OK    %s\n' "$*"; }
bad() {
  printf 'FAIL  %s\n' "$*"
  problems=$((problems + 1))
}
warn() { printf 'WARN  %s\n' "$*"; }

uid=$(id -u)
echo "== Claude Code temp directory"

# Where would Claude Code put its files?
if [ -n "${CLAUDE_CODE_TMPDIR:-}" ]; then
  base=${CLAUDE_CODE_TMPDIR}
  src="CLAUDE_CODE_TMPDIR"
elif [ -n "${TMPDIR:-}" ]; then
  base=${TMPDIR}
  src="TMPDIR"
else
  base=${CKM_ENV_CHECK_DEFAULT_TMP:-/tmp} # the override exists only for testing this script
  src="the default (/tmp)"
fi
note "base temp directory: ${base} (from ${src})"

if [ -L "${base}" ]; then
  warn "${base} is a symlink to $(readlink -f "${base}")"
fi

real=$(readlink -f "${base}" 2> /dev/null || true)
if [ -z "${real}" ] || [ ! -d "${real}" ]; then
  bad "${base} does not exist or is not a directory"
elif [ ! -w "${real}" ]; then
  bad "${real} is not writable by ${USER:-uid ${uid}}"
else
  probe="${real}/.ckm-env-probe-$$"
  if mkdir "${probe}" 2> /dev/null; then
    rmdir "${probe}"
    ok "can create directories under ${real}"
  else
    bad "cannot create a directory under ${real}"
  fi
  # Claude Code appends /claude-<uid>
  per_user="${real}/claude-${uid}"
  if [ -d "${per_user}" ]; then
    ok "per-user directory exists: ${per_user}"
  else
    note "per-user directory ${per_user} does not exist yet (Claude Code creates it)"
  fi
fi

# Suggest a fix when /tmp itself is the problem: a symlink, or not a writable directory.
if [ "${src}" = "the default (/tmp)" ]; then
  tmp_real=$(readlink -f "${base}" 2> /dev/null || true)
  if [ -L "${base}" ] || [ ! -d "${base}" ] || [ ! -w "${base}" ]; then
    echo
    echo "== Suggested fix"
    if [ -n "${tmp_real}" ] && [ -d "${tmp_real}" ] && [ -w "${tmp_real}" ]; then
      target=${tmp_real}
    else
      target="<a writable local directory, for example /dev/shm or a node-local scratch dir>"
    fi
    note "Point Claude Code at the real directory. Pick ONE of these:"
    echo
    note "(a) In your shell startup file (~/.bashrc), before you launch claude:"
    note "      export CLAUDE_CODE_TMPDIR=${target}"
    note "      export TMPDIR=${target}"
    echo
    note "(b) In ~/.claude/settings.json:"
    note '      { "env": { "TMPDIR": "'"${target}"'", "CLAUDE_CODE_TMPDIR": "'"${target}"'" } }'
    echo
    note "If your home directory is shared between clusters, the path above may not exist"
    note "on the others. Then set it only on this machine, for example in ~/.bashrc:"
    note '      [ "$(hostname -s)" = "'"$(hostname -s)"'" ] && export CLAUDE_CODE_TMPDIR='"${target}"
  fi
fi

echo
echo "== Sandbox dependencies"
if command -v bwrap > /dev/null 2>&1; then
  ok "bubblewrap: $(bwrap --version 2>&1 | head -1)"
  # Informational only: this probe can fail when already running inside a sandbox.
  if bwrap --ro-bind / / --dev /dev true > /dev/null 2>&1; then
    ok "bwrap can create a sandbox here"
  else
    warn "bwrap could not create a test sandbox (expected if this shell is already sandboxed;"
    warn "otherwise unprivileged user namespaces may be disabled, so ask your admins)"
  fi
else
  bad "bwrap (bubblewrap) not found in PATH"
fi
if command -v socat > /dev/null 2>&1; then
  ok "socat found"
else
  warn "socat not found (the sandbox needs it for network filtering)"
fi
if command -v git > /dev/null 2>&1; then
  ok "git: $(git --version)"
else
  bad "git not found"
fi

echo
if [ "${problems}" -eq 0 ]; then
  echo "No problems found. If Bash still fails, run 'claude --debug' and look at the"
  echo "[Sandbox Linux] lines."
else
  echo "${problems} problem(s) found; see the messages above."
fi
exit $((problems > 0))
