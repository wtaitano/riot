# Plan: build ckm inside riot on riot's (tensor-train) Parthenon (third sub-PR, stacked on #2)

Generated with Claude Code (agentic AI). Rename to `ckm-fork-pr-<N>.md` once the
internal PR number is known.

## Why the design changed

The riot developers reviewed the original design
(`tmp/ckm_segregation_design.md`, as reflected in PRs #1 and #2; the revised design is in `tmp/ckm_design_v2.md`, not committed). Their feedback
(paraphrased):

- **Isolation was too rigid.** It is fine for ckm to use riot's `src/`, for example
  its utilities.
- **No second Parthenon.** Don't check out separate submodules for ckm. Instead,
  change the branch that riot's own `external/parthenon` submodule points to, on the
  ckm feature branch.
- **Workflow confirmed.** Keep a separate long-lived feature branch, with small
  sub-PRs into it that are merged often. The goal is a history that people can read,
  so "we have any understanding whatsoever of what the robots have done".

PRs #1 and #2 are kept as they are; the user does not rewrite shared history. This
PR changes the design on top of them.

## Decisions (agreed with the ckm lead)

1. **ckm is part of riot's build.**
   - Riot's `CMakeLists.txt` gets `option(RIOT_BUILD_CKM ... OFF)` and an
     `add_subdirectory(conditional_kinetic_miniapp)` after riot's src and tests, so
     ckm can use riot's targets and Catch2.
   - ckm's `CMakeLists.txt` becomes a subproject: no `project()`, no options, and no
     dependency setup of its own.
2. **Riot's `external/parthenon` moves to the tensor-train commit.**
   - Before switching, riot itself (not ckm) was built against tt-cross `51279a2` and
     passed all 18 of its unit tests (Darwin, openmpi 3.1.6 + parallel HDF5).
   - The `.gitmodules` entry gets `branch = lroberts36/tt-cross`, so
     `git submodule update --remote` can bump it on purpose. The SHA stays pinned.
3. **ckm's own submodules are removed.**
   - `conditional_kinetic_miniapp/external/{parthenon,Catch2}` and their `.gitmodules`
     entries are deleted. ckm uses riot's `external/Catch2` (same v3.7.1 commit).
   - `script/setup.sh` and the `ckm-setup` skill are removed.
4. **Isolation becomes one-way.**
   - ckm may use riot's `src/` and riot's targets. Riot must never depend on ckm.
   - `check_boundary.sh` now checks three things:
     - riot→ckm references appear only in the four wiring files (`.gitmodules`,
       `CMakeLists.txt`, `.github/CODEOWNERS`, `.github/workflows/ckm.yml`);
     - the CMake hook stays off by default;
     - there are no ckm-owned submodules.
   - Not reusing riot *physics* packages remains a guideline in `AGENTS.md` and the
     reviewer agent. It is not a CI check, because it is a case-by-case judgement.
5. **CI builds only the ckm targets.** `ckm.yml` builds `ckm` and `ckm_unit_tests`
   inside riot with `RIOT_BUILD_CKM=ON`. Riot's full suite stays in `unit-tests.yml`,
   which now also exercises riot on the tensor-train Parthenon.

## Contents

- **Riot files:**
  - `.gitmodules`: ckm entries removed; `branch =` added for parthenon.
  - `external/parthenon`: `2ddc3e3` → `51279a2`.
  - `CMakeLists.txt`: `RIOT_BUILD_CKM` option and subdirectory.
  - `.github/workflows/ckm.yml`: in-riot targeted build, new path triggers, and a
    ccache key bump.
- **ckm files:**
  - `CMakeLists.txt` (subproject).
  - `tst/CMakeLists.txt` (input-deck path).
  - `script/check_boundary.sh` (one-way rule); `script/setup.sh` removed.
  - `.claude/`: the `guard_edit` allowlist gains `CMakeLists.txt`; `settings.json`
    allowlist updated; the build, test and format skills are rewritten; `ckm-setup`
    is removed; the agents are updated.
  - `AGENTS.md`, `README.md`, and `doc/sphinx/` (index, building, contributing,
    maintainers).

## Verification

- **Riot on the tensor-train Parthenon** (Darwin, openmpi 3.1.6 + hdf5 1.12.2, gcc
  12.2): build OK, riot unit tests 18/18 pass.
- **Riot + ckm with `RIOT_BUILD_CKM=ON`, MPI:** build OK. `ctest` passes 20/20 (riot's
  18 plus ckm's unit test and driver test). A 4-rank `mpirun` of `ckm` prints `PASS`.
- **Riot + ckm, serial** (gcc 12.2, system serial HDF5): the `ckm` and
  `ckm_unit_tests` targets build, and ckm's tests pass 2/2.
- **Known problem, already on riot `main`:** riot's `riotlib` does not compile with
  `RIOT_ENABLE_MPI=OFF`. `riot_utils.hpp` and `laser.cpp` use `MPI_Comm`, `MPI_Bcast`
  and `MPI_Allreduce` without a guard. This was reproduced on riot's original
  Parthenon pin `2ddc3e3`, so neither ckm nor the tensor-train branch causes it.
  Riot's CI never builds serial. For now, serial ckm works only while `ckmlib` does
  not link `riotlib`. This is documented in `AGENTS.md`, `building.rst`, and the
  `ckm-build` skill.
- **Other checks:** `check_boundary.sh` passes; the docs build with `-W -n`;
  `ckm.yml` parses; clang-format 20 is clean.

## Follow-up: submodule fetch is a human step under the sandbox

- **Finding.** The shared sandbox settings keep `../.git/config` read-only (on
  purpose: code configured there runs outside the sandbox). Every git command that
  writes it therefore fails inside the sandbox with `could not lock config file`:
  `git submodule update --init`, `git push -u`, `git branch --track`, and
  `git worktree add`/`prune`. Commit, `switch -c --no-track`, and HTTPS fetch work.
- **Decision (agreed with the ckm lead).** Keep `config` protected and make the
  submodule fetch a one-time human step, run in a normal shell. The rejected
  alternative was `sandbox.excludedCommands: ["git submodule *"]`. It runs every
  `git submodule` command with full access (`git submodule foreach` is an arbitrary
  unsandboxed shell). It also matches only the bare command, not `git -C ..` or `&&`
  chains, and admin-required sandboxes ignore repository `excludedCommands`.
- **Changes.**
  - The submodule command is removed from Claude's build instructions (`AGENTS.md`,
    the `ckm-build` skill) and from the `settings.json` allowlist.
  - Agents now check `git submodule status` and ask the human through
    `scratch/commands.txt`.
  - `README.md`, `building.rst`, `contributing.rst` and `maintainers.rst` say to run
    it outside Claude.
- **Verification.** `check_boundary.sh` passes; the docs build with `-W -n`;
  `settings.json` parses.
