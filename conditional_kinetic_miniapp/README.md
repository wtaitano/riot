# Conditional kinetic miniapp (ckm)

ckm is a [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) application
developed inside riot on a long-lived feature branch, `wtaitano/ckm`. It is built
as part of riot with `-DRIOT_BUILD_CKM=ON`. It uses riot's own `external/parthenon`;
on the feature branch that submodule points at the tensor-train branch of Parthenon
(block-AMR tensor-train library, Parthenon PR
[#1464](https://github.com/parthenon-hpc-lab/parthenon/pull/1464)).

This file was made in part with generative AI.

## Isolation rules

- **riot never depends on ckm.** No file outside this directory references ckm,
  except the files that wire it in: `.gitmodules`, `CMakeLists.txt` (the
  off-by-default `RIOT_BUILD_CKM` option), `.github/CODEOWNERS`, and
  `.github/workflows/ckm.yml`. `script/check_boundary.sh` enforces this in CI.
- **ckm may use riot.** It can use riot's `src/` utilities and build targets. It does
  not reuse riot's physics packages; the ckm team writes its own.
- **No dependency copies.** ckm has no submodules of its own. It uses riot's
  `external/` (Parthenon, Kokkos, Catch2).
- **Process and tooling are riot's.** That includes formatting (`script/format.sh`,
  `.clang-format`), the PR template, and the
  [contributing guide](../doc/sphinx/src/contributing.rst) with its AI-assisted
  coding policy.

## Building

Run everything from the riot root. ckm needs C++20 (required by the tensor-train
branch), CMake 3.26 or newer, and HDF5. MPI is on by default, and with MPI on, HDF5
must be parallel.

```bash
git submodule update --init --recursive \
  external/parthenon external/kokkos-kernels external/singularity-eos external/singularity-opac

cmake -S . -B build-ckm -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON -DRIOT_BUILD_PYTHON=OFF
cmake --build build-ckm -j --target ckm ckm_unit_tests
ctest --test-dir build-ckm/conditional_kinetic_miniapp --output-on-failure
```

For a serial build, add `-DRIOT_ENABLE_MPI=OFF` and use a serial HDF5. All of riot's
options (`RIOT_ENABLE_MPI`, `RIOT_ENABLE_CUDA`, `RIOT_BUILD_CATCH2`, ...) apply to
ckm as well. On Darwin:

```bash
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3   # MPI
module load gcc/12.2.0 cmake/3.26.3   # serial; add -DHDF5_ROOT=/usr for serial HDF5
```

## Running

```bash
build-ckm/conditional_kinetic_miniapp/src/ckm -i conditional_kinetic_miniapp/inputs/skeleton.in
```

The skeleton driver fills a tensor-train field on every mesh block and checks the
result. On success it prints `ckm: tensor-train check on N blocks: PASS`.

## Layout

```
CMakeLists.txt     ckm subproject (added by riot when RIOT_BUILD_CKM=ON)
src/               ckm library (ckmlib) and the ckm executable
tst/               Catch2 unit tests and driver tests
inputs/            input decks
script/            isolation check, contributor and maintainer scripts
doc/sphinx/        ckm documentation
plan_histories/    agentic-AI plan files (riot policy)
AGENTS.md          instructions for AI coding agents (CLAUDE.md imports it)
.claude/           shared Claude Code settings, hooks, agents, skills
```

## AI coding agents

Read [`AGENTS.md`](AGENTS.md). Claude Code users should launch `claude` from **this
directory**, because the shared `.claude/` settings, hooks, agents and skills only
apply there. See the "AI coding agents" section of `doc/sphinx/src/contributing.rst`.

## Contributing

ckm is developed on the `wtaitano/ckm` branch of the
[wtaitano/riot](https://github.com/wtaitano/riot) fork, through small sub-PRs that
are merged often. It goes into `lanl/riot` as one PR at the end. See
`doc/sphinx/src/contributing.rst` for the workflow.
