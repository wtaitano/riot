---
name: ckm-build
description: Configure and build ckm as part of riot (RIOT_BUILD_CKM=ON), serial or MPI (default both). Use whenever ckm sources or CMake change, before running tests, or when asked to build.
---

<!-- This file was made in part with generative AI. -->

ckm is built inside riot's CMake project. Run everything from the **riot root**
(`cd ..` from the miniapp). Build directories go there as `build-ckm-*`, which riot's
`.gitignore` already ignores.

**Submodules.** Do this once, and again after a Parthenon pin bump:

```bash
git submodule update --init --recursive \
  external/parthenon external/kokkos-kernels external/singularity-eos external/singularity-opac
```

**MPI** (the default):

```bash
module load openmpi/3.1.6-gcc_12.2.0 hdf5/1.12.2 cmake/3.26.3 && \
cmake -S . -B build-ckm-mpi -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
  -DRIOT_BUILD_PYTHON=OFF && \
cmake --build build-ckm-mpi -j 32 --target ckm ckm_unit_tests
```

**Serial:**

```bash
module load gcc/12.2.0 cmake/3.26.3 && \
cmake -S . -B build-ckm-serial -G Ninja -DRIOT_BUILD_CKM=ON -DRIOT_ENABLE_UNIT_TESTS=ON \
  -DRIOT_BUILD_PYTHON=OFF -DRIOT_ENABLE_MPI=OFF -DHDF5_ROOT=/usr && \
cmake --build build-ckm-serial -j 32 --target ckm ckm_unit_tests
```

## Notes

- **Modules.** Load modules in the **same shell command** as `cmake`. Separate tool
  calls do not keep loaded modules.
- **Compiler.** The system gcc 8.5 is too old. Always load `gcc/12.2.0`, either
  directly or through the openmpi module.
- **Serial HDF5.** The `hdf5/1.12.2` module is *parallel*, so a serial build uses the
  system serial HDF5 (`-DHDF5_ROOT=/usr`).
- **Serial and `riotlib`.** Riot's `riotlib` does not compile without MPI; this
  problem is also on riot `main`. In serial, build **only** the ckm targets. If a
  serial build fails inside `src/` (riot's code), that is why.
- **Targets.** Build only `ckm ckm_unit_tests`. If you changed riot files (the root
  `CMakeLists.txt` or the Parthenon pin), also build the default target and run riot's
  tests (`ctest --test-dir build-ckm-mpi`).
- **Output.** Send build output to a log file and show only errors and the last few
  lines.

Report pass or fail for each configuration, with the first compiler error if a build
fails.
