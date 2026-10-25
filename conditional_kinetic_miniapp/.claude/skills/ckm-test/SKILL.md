---
name: ckm-test
description: Run the ckm test suite (Catch2 unit tests + driver run test) for the serial and MPI builds, plus a multi-rank driver run. Use after ckm-build, before reporting any change as done.
---

<!-- This file was made in part with generative AI. -->

Run these from the riot root, after `ckm-build`. The ckm tests live under the
`conditional_kinetic_miniapp` subdirectory of each build directory.

```bash
# Serial
module load gcc/12.2.0 cmake/3.26.3 && \
ctest --test-dir build-ckm-serial/conditional_kinetic_miniapp --output-on-failure

# MPI: ctest plus a 4-rank driver run
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3 && \
ctest --test-dir build-ckm-mpi/conditional_kinetic_miniapp --output-on-failure && \
(cd build-ckm-mpi/conditional_kinetic_miniapp/tst && timeout 300 mpirun --oversubscribe \
  -np 4 ../src/ckm -i ../../../conditional_kinetic_miniapp/inputs/skeleton.in)
```

## Running a single test

- Unit tests:
  `build-ckm-*/conditional_kinetic_miniapp/tst/unit/ckm_unit_tests "<test name>"`.
  You can also filter by tag, for example `"[tt]"`.
- Driver test: `ctest -R ckm_skeleton_run`.
- Riot's own tests: if you changed riot wiring files, also run
  `ctest --test-dir build-ckm-mpi`.

## Report

Give the pass count for each configuration, the driver's `PASS`/`FAIL` line, and the
full output of any failing test. Never report success while a test is failing or was
skipped.
