# AGENTS.md: conditional kinetic miniapp (ckm)

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, ...) and for
humans working with them. `CLAUDE.md` imports this file. This file was made in part
with generative AI.

## Where to start

Launch your agent **from `conditional_kinetic_miniapp/`**, not from the riot root. The
shared Claude Code settings, hooks, agents and skills in `.claude/` only apply when
the session starts here.

## What ckm is

ckm is a Parthenon application developed inside riot on a long-lived feature branch
(`wtaitano/ckm`).

- **Build.** It is built as part of riot with `-DRIOT_BUILD_CKM=ON`.
- **Parthenon.** It uses riot's own `external/parthenon`. On the feature branch that
  submodule points at the tensor-train branch (`lroberts36/tt-cross`, Parthenon PR
  #1464).
- **Names.** All code lives in namespace `ckm`. The executable is `ckm` and the
  library is `ckmlib`.
- **More detail.** See `README.md` and `doc/sphinx/`.

## Isolation rules

1. **Riot must never depend on ckm.** Do not edit riot files to include, link, or call
   ckm. You may edit files outside `conditional_kinetic_miniapp/` only to wire ckm in,
   and only these: `../.gitmodules`, `../CMakeLists.txt` (the `RIOT_BUILD_CKM`
   option), `../.github/CODEOWNERS`, and `../.github/workflows/ckm.yml`.
   `script/check_boundary.sh` enforces this, and CI runs it.
2. **ckm may use riot.**
   - **Utilities** from `../src/` are fine. Self-contained headers such as
     `riot_utils/type_maps.hpp` can be included directly. Anything that needs riot's
     variables or packages means linking `riotlib`.
   - **Physics packages:** do not reuse riot's (hydro, radiation, materials, ...).
     ckm writes its own. If reusing one seems worthwhile, ask the human first.
3. **No dependency copies.** ckm has no submodules of its own. It uses riot's
   `../external/` (Parthenon, Kokkos, Catch2).

Tooling and process are riot's: `../script/format.sh`, `../.clang-format`, the PR
template, and the AI policy below.

## Build, test, format

Run everything from `conditional_kinetic_miniapp/`.

- **Source.** The CMake source is the riot root (`-S ..`).
- **Build directories.** They live **inside the miniapp** as `build-mpi/` and
  `build-serial/`, so they stay writable under the Claude Code sandbox, which only
  allows writes in the launch directory. Riot's `.gitignore` rule `build*/` already
  ignores them.
- **Modules.** Load modules **in the same shell command** as the build, and never
  pipe `module load` (for example `module load ... | head`). The pipe runs it in a
  subshell, so nothing gets loaded and no error is printed. `module load` works
  inside the sandbox.
- **Submodules are a human step.** `git submodule update --init` must write
  `../.git/config`, which the sandbox keeps read-only, so it fails inside the sandbox
  with `could not lock config file`. The human runs it outside Claude (README "Getting
  started", step 5), once per clone and again after a submodule pin changes. If
  `../external/parthenon/CMakeLists.txt` is missing or `git -C .. submodule status`
  shows a `-` or `+` in front of a path, do not run it yourself and do not retry it
  unsandboxed. Write the command to `scratch/commands.txt` (see "Pushing" under Git
  workflow) and ask the human to run it.

```bash
# Darwin, MPI (default):
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3 && \
cmake -S .. -B build-mpi -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
  -DRIOT_BUILD_PYTHON=OFF && \
cmake --build build-mpi -j 32 --target ckm ckm_unit_tests && \
ctest --test-dir build-mpi/conditional_kinetic_miniapp --output-on-failure

# Darwin, serial: also pass -DRIOT_ENABLE_MPI=OFF -DHDF5_ROOT=/usr
module load gcc/12.2.0 cmake/3.26.3 && \
cmake -S .. -B build-serial -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
  -DRIOT_BUILD_PYTHON=OFF -DRIOT_ENABLE_MPI=OFF -DHDF5_ROOT=/usr

# Format check of ckm C++ (clang-format 20, riot's config):
module load clang/20.1.8 && git ls-files -co --exclude-standard '*.cpp' '*.hpp' \
  | xargs -r clang-format --dry-run --Werror
```

**Sandbox.** The shared `.claude/settings.json` gives sandboxed commands write
access to the repo's `../.git`, so `git add`/`git commit` work. `../.git/hooks` and
`../.git/config` stay write-protected, because code written there would run outside
the sandbox. Do not try to work around that.

**Network.** The sandbox allows only github.com (submodules, the forks, and the API)
and PyPI (for the docs build). Anything else prompts the human. SSH is not available
inside the sandbox, because `~/.ssh` is hidden. Fetch over HTTPS from the `ckm` and
`upstream` remotes. **Pushing is always done by a human, outside the agent.**

- **Requirements:** C++20 (required by the tensor-train branch), CMake 3.26 or newer,
  and HDF5. ckm refuses `RIOT_ENABLE_HDF5=OFF`.
- **Targets.** Build only `ckm ckm_unit_tests` unless you changed riot files. A full
  riot build takes much longer.
- **Serial builds and `riotlib`.** Riot's own `riotlib` does **not** compile with
  `RIOT_ENABLE_MPI=OFF`; this is a problem on riot `main` too, not caused by ckm. The
  serial ckm build works only while `ckmlib` does not link `riotlib`. If you need a
  riot utility that comes through `riotlib`, ask the human first, because it would
  make ckm MPI-only until riot fixes its serial build.
- **If you touched a riot file** (one of the wiring files above), also build `riot`
  and run riot's unit tests.
- **Before you report a change as done:** serial and MPI both build, `ctest` passes in
  both, and `conditional_kinetic_miniapp/script/check_boundary.sh` passes.

## Code conventions

- C++20, Kokkos/Parthenon idioms, and `parthenon::Real` for floating point.
- Every new source file gets riot's copyright header (`../script/format.sh` adds it)
  and the line `This file was made in part with generative AI.` if an agent wrote any
  of it.
- New unit tests go in `tst/unit/` and must be added to the source list in
  `tst/unit/CMakeLists.txt`. Use Catch2 v3 headers (`<catch2/catch_test_macros.hpp>`)
  and the `GIVEN`/`WHEN`/`THEN` style.
- New sources go in `src/CMakeLists.txt` `SRC_LIST`, in alphabetical order.
- Tensor trains have reference semantics. Hold them through
  `std::shared_ptr<TensorTrain>` and avoid copies. Read
  `../external/parthenon/doc/sphinx/src/tensor_trains.rst` before touching TT code.

## AI-assisted coding policy (riot's, unchanged)

The policy is riot's
[`doc/sphinx/src/contributing.rst`](../doc/sphinx/src/contributing.rst), section
"AI-assisted coding":

- **Disclaimer.** Add a disclaimer comment to every file an agent helped write.
- **Plan file.** For agentic work, write the plan to `plan_histories/`, named after
  the **internal fork PR number** as `plan_histories/ckm-fork-pr-<N>.md`. Use a
  descriptive name until the number is known.
- **Ownership.** The human who submits the PR owns the code and must understand it.
- **Size.** Keep each change small enough to review. Riot's developers asked for
  frequent, small sub-PRs into the feature branch so the history stays readable.

## Git workflow

- **Feature branch.** ckm is developed on `wtaitano/ckm` on the `wtaitano/riot` fork,
  and merged into `lanl/riot` as one final PR.
- **Sub-PRs.** Each topic branch, `<github-user>/ckm-<topic>`, becomes a sub-PR into
  `wtaitano/riot:wtaitano/ckm`, **never** into `lanl/riot`. Merge sub-PRs often. If a
  review is pending, stack the next PR on the previous branch.
- **Remotes.** `origin` is your fork. `ckm` and `upstream` are fetch-only. Set them up
  with `script/setup_remotes.sh <you>`.
- **Never rewrite shared history.** Do not rebase, force-push, or reset
  `wtaitano/ckm` or `main`. Bring in riot changes with `git merge upstream/main`.
  Rebasing your own unmerged topic branch is fine.
- **Merges.** Merge commits only, with no squash or rebase merges. `git pull` must not
  rebase (`pull.rebase=false`, `pull.ff=only`).
- **Pushing.** Agents commit locally. A human decides when to push and when to open a
  PR. When you need a human to run commands (a push, a PR, anything outside the
  sandbox), write them to **`scratch/commands.txt`**, holding only the next runnable
  step, and give the full command including the `cd`.

## Scratch area

`scratch/` (git-ignored) is the place for anything personal or temporary: commands a
human must run, notes, drafts, and throwaway outputs. It is inside the miniapp, so the
sandbox and the edit guard both allow writes there. It exists only on your machine, so
anything the team should see goes in `plan_histories/` or `doc/` instead. Do not write
scratch files in the riot checkout outside the miniapp, because the guards block it.

## If Bash fails with `bwrap: Can't mkdir /tmp/claude-empty-...`

Claude Code's sandbox cannot create its temp directory. This happens on machines where
`/tmp` is a symlink (for example Darwin, where `/tmp` points to `/ram/tmp`) or is not
writable. It fails even with the sandbox switched off.

1. Run `script/check_env.sh`. It checks the temp directory and the sandbox
   dependencies, and prints the exact lines to use on **this** machine.
2. Set `CLAUDE_CODE_TMPDIR` (and `TMPDIR`) to the real directory it reports, in your
   shell startup file or in `~/.claude/settings.json`, then restart `claude`.

This cannot be committed to the repo, because Claude Code ignores
`CLAUDE_CODE_TMPDIR` in project settings. The right path differs per machine. On
Darwin it is `/ram/tmp`. If your home directory is shared between clusters, set it
only on the machine that needs it.

## Pointers

- `README.md`: overview and layout.
- `doc/sphinx/src/building.rst`, `contributing.rst`, `maintainers.rst`.
- `plan_histories/`: past agent plans, including why the design changed after riot
  review (`ckm-shared-parthenon.md`).
- `.claude/`: shared Claude Code settings, hooks, agents (`ckm-implementer`,
  `ckm-reviewer`), and skills (`ckm-build`, `ckm-test`, `ckm-format`,
  `ckm-plan-history`).
