# Plan: ckm skeleton (wtaitano/riot#1, first internal PR into wtaitano/ckm)

Generated with Claude Code (agentic AI).

## Goal

Stand up the conditional kinetic miniapp (ckm) as a standalone Parthenon application
inside riot, isolated from riot's code and build, using the tensor-train Parthenon
branch. No physics.

## Decisions (agreed with the ckm lead before implementation)

- Zero shared source with riot. Process and tooling (format, PR template, AI policy)
  are inherited from the riot root; code and build are not.
- Separate Parthenon submodule at `conditional_kinetic_miniapp/external/parthenon`,
  public upstream URL, pinned to `51279a2` (`lroberts36/tt-cross`, PR #1464). That
  commit is 89 commits ahead of and 0 behind riot's Parthenon pin `2ddc3e3`, with the
  same Kokkos (`267ebc2`). The pin moves to a tag once one exists.
- Separate Catch2 submodule (v3.7.1, same commit as riot's).
- Both submodules are `update = none`, so riot users and riot CI never fetch them.
  `script/setup.sh` fetches them on purpose.
- HDF5 is required. `CKM_ENABLE_MPI` defaults ON with riot's parallel-HDF5 check.
  C++20 is required, because the tensor-train branch uses concepts.
- Standalone CMake only; riot's top-level CMake is unchanged.
- The namespace, executable, and option prefix are `ckm` / `CKM_*`.
- Root files touched: `.gitmodules`, `.github/CODEOWNERS`, and
  `.github/workflows/ckm.yml` (path-filtered CI). `script/check_boundary.sh` allows
  exactly these to reference the miniapp.
- Workflow: a long-lived `wtaitano/ckm` branch on the fork, internal PRs into it, and
  one final PR to `lanl/riot`.

## Contents

- `CMakeLists.txt`, `src/` (`ckmlib`, `ckm` driver), `tst/` (Catch2 unit test plus a
  driver run test), and `inputs/skeleton.in`.
- The `ckm` driver registers one TT field per block, fills it on device, and checks
  the sum.
- `script/setup.sh`, `script/check_boundary.sh`, `script/setup_remotes.sh`, and
  `script/github/setup_fork_repo.sh` (fork lockdown via `gh api`).
- Docs: `doc/sphinx` with index, building, contributing, and maintainers pages.

## Verification

- Darwin, gcc 12.2.0, serial (system HDF5 1.10.5): build, `ctest` 2/2 pass, and the
  driver passes.
- Darwin, `openmpi/3.1.6-gcc_12.2.0` with `hdf5/1.12.2` (parallel): build, `ctest`
  2/2 pass, and `mpirun -np 4` driver passes.
- `check_boundary.sh` passes. Lockdown script dry-run against a stub `gh` (JSON
  validated).
