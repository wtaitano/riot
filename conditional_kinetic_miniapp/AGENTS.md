# AGENTS.md: conditional kinetic miniapp (ckm)

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, ...) and for
humans working with them. `CLAUDE.md` imports this file. This file was made in part
with generative AI.

## Where to start

Launch your agent **from `conditional_kinetic_miniapp/`**, not from the riot root. The
shared Claude Code settings, hooks, agents and skills in `.claude/` only apply when
the session starts here.

## What ckm is

ckm is a standalone Parthenon application inside the riot repository. It is built on
the tensor-train branch of Parthenon (`external/parthenon`, PR #1464). All code lives
in namespace `ckm`. The executable is `ckm` and the library is `ckmlib`. Build options
are named `CKM_*`. See `README.md` and `doc/sphinx/`.

## Isolation rules (hard requirements)

1. **Never** include, link, or reference riot's `../src/`, `../external/`, `../tst/`,
   `riotlib`, `riot::`, or `RIOT_*` from ckm code or CMake. ckm writes its own
   physics; do not port or wrap riot's physics packages.
2. **Never** edit files outside `conditional_kinetic_miniapp/`. The only exceptions are
   `../.gitmodules`, `../.github/CODEOWNERS`, and `../.github/workflows/ckm.yml`, and
   only when the task requires them.
3. Run `script/check_boundary.sh` before reporting work as done. CI runs it too.

Tooling and process are inherited from riot, and you should use them: riot's
`../script/format.sh`, `../.clang-format`, the PR template, and the AI policy below.

## Build, test, format

```bash
script/setup.sh                       # fetch ckm submodules (once, and after pin bumps)

# Darwin, MPI (default):
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3
cmake -S . -B ../build-ckm -G Ninja -DCKM_ENABLE_UNIT_TESTS=ON
cmake --build ../build-ckm -j 32
ctest --test-dir ../build-ckm --output-on-failure

# Darwin, serial:
module load gcc/12.2.0 cmake/3.26.3
cmake -S . -B ../build-ckm-serial -G Ninja -DCKM_ENABLE_MPI=OFF \
  -DCKM_ENABLE_UNIT_TESTS=ON -DHDF5_ROOT=/usr

# Format: riot's script runs clang-format 20 and black over the whole repo
# (black must be on PATH), from the repo root:
module load clang/20.1.8 && (cd .. && ./script/format.sh)
# Quick check of ckm C++ only, without black:
git ls-files '*.cpp' '*.hpp' | grep -v '^external/' | xargs clang-format --dry-run --Werror
```

- **Requirements:** C++20 (required by the TT branch), CMake 3.26 or newer, and HDF5,
  which is required. With MPI on, HDF5 must be parallel.
- **Build directories** go at the riot root as `../build-ckm*`. Riot's `.gitignore`
  already ignores `build*/`.
- **Before you report a change as done:** it builds in both serial and MPI, `ctest`
  passes in both, and `script/check_boundary.sh` passes.

## Code conventions

- C++20, Kokkos/Parthenon idioms, and `parthenon::Real` for floating point.
- Every new source file gets riot's copyright header (`../script/format.sh` adds it)
  and the line `This file was made in part with generative AI.` if an agent wrote any
  of it.
- New unit tests go in `tst/unit/`, added to the source list in
  `tst/unit/CMakeLists.txt`. Use Catch2 v3 headers (`<catch2/catch_test_macros.hpp>`)
  and the `GIVEN`/`WHEN`/`THEN` style.
- New sources go in `src/CMakeLists.txt` `SRC_LIST`, in alphabetical order.
- Tensor trains: hold them through `std::shared_ptr<TensorTrain>` and avoid copies
  (they have reference semantics). Read
  `external/parthenon/doc/sphinx/src/tensor_trains.rst` before touching TT code.

## AI-assisted coding policy (riot's, unchanged)

The policy is riot's
[`doc/sphinx/src/contributing.rst`](../doc/sphinx/src/contributing.rst), section
"AI-assisted coding":

- **Disclaimer.** Add a disclaimer comment to every file an agent helped write.
- **Plan file.** For agentic work, write the plan to `plan_histories/`. Name it after
  the **internal fork PR number**, as `plan_histories/ckm-fork-pr-<N>.md`. Use a
  descriptive name until the number is known, then rename it.
- **Ownership.** The human who submits the PR owns and must understand the code.
- **Size.** Keep changes reviewable. Prefer several small PRs over one large one.

## Git workflow

- **Integration branch.** ckm is developed on `wtaitano/ckm` on the `wtaitano/riot`
  fork, and merged into `lanl/riot` as one final PR.
- **Topic branches.** Name topic branches `<github-user>/ckm-<topic>`. Open PRs into
  `wtaitano/riot:wtaitano/ckm`, **never** into `lanl/riot`.
- **Remotes.** `origin` is your fork. `ckm` and `upstream` are fetch-only. Set them up
  with `script/setup_remotes.sh <you>`.
- **Never rewrite shared history.** Do not rebase, force-push, or reset
  `wtaitano/ckm` or `main`. Riot changes come in with `git merge upstream/main`.
  Rebasing your own unmerged topic branch is fine.
- **Merges.** Merge commits only (no squash or rebase merges). `git pull` must not
  rebase (`pull.rebase=false`, `pull.ff=only`).
- **Push and PRs.** Agents commit locally. A human decides when to push and when to
  open a PR.

## Pointers

- `README.md`: overview, options, and layout.
- `doc/sphinx/src/building.rst`, `contributing.rst`, `maintainers.rst`.
- `plan_histories/`: past agent plans.
- `.claude/`: shared Claude Code settings, hooks, agents (`ckm-implementer`,
  `ckm-reviewer`), and skills (`ckm-setup`, `ckm-build`, `ckm-test`, `ckm-format`,
  `ckm-plan-history`).
