# Conditional kinetic miniapp (ckm)

A standalone [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) application
that lives inside the riot repository but shares **no source or build** with riot. It
uses the tensor-train branch of Parthenon (block-AMR tensor-train library, Parthenon
PR [#1464](https://github.com/parthenon-hpc-lab/parthenon/pull/1464)), pinned in
`external/parthenon`.

This file was made in part with generative AI.

## Isolation rules

- Nothing under `conditional_kinetic_miniapp/` includes, links, or builds anything from
  riot's `src/`, `external/`, or physics packages.
- Nothing outside this directory references it, except `.gitmodules`,
  `.github/CODEOWNERS`, and `.github/workflows/ckm.yml`, which git and GitHub require at
  the repository root.
- `script/check_boundary.sh` enforces both directions in CI.
- Process and tooling are inherited from riot: formatting (`script/format.sh`,
  `.clang-format`), the PR template, and the
  [contributing guide](../doc/sphinx/src/contributing.rst), including its AI-assisted
  coding policy.

## Getting the code

The ckm submodules are marked `update = none`, so `git clone --recursive` of riot does
not fetch them. Fetch them explicitly:

```bash
conditional_kinetic_miniapp/script/setup.sh
```

## Building

ckm requires a C++20 compiler, CMake >= 3.26, and HDF5. MPI is on by default; with
MPI on, HDF5 must be parallel.

```bash
# MPI (default), parallel HDF5
cmake -S conditional_kinetic_miniapp -B build-ckm -DCKM_ENABLE_UNIT_TESTS=ON
cmake --build build-ckm -j
ctest --test-dir build-ckm --output-on-failure

# Serial, serial HDF5
cmake -S conditional_kinetic_miniapp -B build-ckm-serial \
  -DCKM_ENABLE_MPI=OFF -DCKM_ENABLE_UNIT_TESTS=ON
```

On Darwin:

```bash
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3       # MPI
module load gcc/12.2.0 cmake/3.26.3   # serial; add -DHDF5_ROOT=/usr for serial HDF5
```

| Option | Default | Meaning |
|---|---|---|
| `CKM_ENABLE_MPI` | `ON` | Build with MPI (requires parallel HDF5) |
| `CKM_ENABLE_UNIT_TESTS` | `OFF` | Build the Catch2 unit tests and the driver run test |
| `CKM_BUILD_CATCH2` | `ON` | Use the bundled Catch2; `OFF` uses a system Catch2 v3 |
| `CKM_ENABLE_CUDA` | `OFF` | Build for CUDA |
| `CKM_ENABLE_OPENMP` | `OFF` | Build with OpenMP |
| `CKM_ENABLE_WARNINGS` | `OFF` | Enable compiler warnings |

## Running

```bash
build-ckm/src/ckm -i conditional_kinetic_miniapp/inputs/skeleton.in
```

The skeleton driver fills a tensor-train field on every mesh block and checks the
result. It prints `ckm: tensor-train check on N blocks: PASS`.

## Layout

```
CMakeLists.txt     standalone project (ckm)
src/               ckm library (ckmlib) and the ckm executable
tst/               Catch2 unit tests and driver tests
inputs/            input decks
script/            setup, isolation check, contributor and maintainer scripts
doc/sphinx/        ckm documentation
plan_histories/    agentic-AI plan files (riot policy)
external/          parthenon (tensor-train branch), Catch2
AGENTS.md          instructions for AI coding agents (CLAUDE.md imports it)
.claude/           shared Claude Code settings, hooks, agents, skills
```

## AI coding agents

Read [`AGENTS.md`](AGENTS.md). Claude Code users should launch `claude` from **this
directory**, because the shared `.claude/` settings, hooks, agents, and skills only
apply there. See the "AI coding agents" section of `doc/sphinx/src/contributing.rst`.

## Contributing

ckm is developed on the `wtaitano/ckm` branch of the
[wtaitano/riot](https://github.com/wtaitano/riot) fork, and merged into `lanl/riot`
as one PR at the end. See `doc/sphinx/src/contributing.rst` for the workflow.
