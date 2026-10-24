---
name: ckm-build
description: Configure and build ckm, serial or MPI (default both). Use whenever ckm sources or CMake change, before running tests, or when asked to build.
---

<!-- This file was made in part with generative AI. -->

Run from `conditional_kinetic_miniapp/`. Build directories live at the riot root as
`../build-ckm*`; riot's `.gitignore` already ignores them. Use Ninja and `-j 32`.

**MPI** (the default configuration):

```bash
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3
cmake -S . -B ../build-ckm-mpi -G Ninja -DCKM_ENABLE_UNIT_TESTS=ON
cmake --build ../build-ckm-mpi -j 32
```

**Serial:**

```bash
module load gcc/12.2.0 cmake/3.26.3
cmake -S . -B ../build-ckm-serial -G Ninja -DCKM_ENABLE_MPI=OFF \
  -DCKM_ENABLE_UNIT_TESTS=ON -DHDF5_ROOT=/usr
cmake --build ../build-ckm-serial -j 32
```

## Notes

- **Modules.** Load the modules and run `cmake` **in the same shell command**.
  Separate tool calls do not keep loaded modules.
- **gcc 8.5 is too old.** The system gcc 8.5 cannot build C++20 Parthenon. Always load
  `gcc/12.2.0`, either directly or through the openmpi module.
- **Serial HDF5.** The `hdf5/1.12.2` module is *parallel*, so a serial build must use
  the system serial HDF5 (`-DHDF5_ROOT=/usr`).
- **Output.** Send build output to a log file and show only errors and the last
  lines. The Parthenon build is long and noisy.
- **First build.** Expect about 1 minute with `-j 48`. Incremental builds are seconds.
- **Off Darwin.** Use the commands in `doc/sphinx/src/building.rst`.

Report pass or fail for each configuration, with the first compiler error if there is
one.
