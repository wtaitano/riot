---
name: ckm-setup
description: Fetch or refresh the ckm submodules (tensor-train Parthenon + Kokkos, Catch2). Use on a fresh clone, after pulling a Parthenon pin bump, or when CMake says the ckm Parthenon submodule is not checked out.
---

<!-- This file was made in part with generative AI. -->

Run from `conditional_kinetic_miniapp/`:

```bash
script/setup.sh
```

It checks out `external/parthenon` (and Parthenon's own submodules, including Kokkos)
and `external/Catch2` at their pinned commits. The ckm submodules are marked
`update = none` in riot's `.gitmodules`, so a plain
`git submodule update --init --recursive` deliberately skips them. Use this script
instead.

Report the final `git submodule status` lines (pinned SHAs). If a pin changed, a
fresh build directory is safest (`rm -rf ../build-ckm*`).
