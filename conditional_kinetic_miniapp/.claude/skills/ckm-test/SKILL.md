---
name: ckm-test
description: Run the ckm test suite (Catch2 unit tests + driver run test) for the serial and MPI builds, plus a multi-rank driver run. Use after ckm-build, before reporting any change as done.
---

<!-- This file was made in part with generative AI. -->

Run from `conditional_kinetic_miniapp/`, after building with `ckm-build`.

```bash
# Serial
module load gcc/12.2.0 cmake/3.26.3
ctest --test-dir ../build-ckm-serial --output-on-failure

# MPI: ctest plus a 4-rank driver run
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3
ctest --test-dir ../build-ckm-mpi --output-on-failure
(cd ../build-ckm-mpi/tst && timeout 300 mpirun --oversubscribe -np 4 \
  ../src/ckm -i ../../conditional_kinetic_miniapp/inputs/skeleton.in)
```

## Single tests

- **Unit tests:** run one directly with
  `../build-ckm-*/tst/unit/ckm_unit_tests "<test name>"`, or filter by tag with
  `"[tt]"`.
- **Driver test:** `ctest -R ckm_skeleton_run`.

Report the pass count for each configuration, the driver's `PASS`/`FAIL` line, and the
full output of any failing test. Never report success with a failing or skipped test.
