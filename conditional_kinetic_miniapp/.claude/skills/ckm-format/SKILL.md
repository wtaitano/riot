---
name: ckm-format
description: Format ckm C++ with riot's clang-format 20 config and check it is clean; also check the isolation boundary. Use before committing any ckm change.
---

<!-- This file was made in part with generative AI. -->

Riot's CI (`format.yml`) runs `../script/format.sh` over every tracked `.cpp`/`.hpp`/
`.py` file in the repo, including ckm's. Match it locally. Run from
`conditional_kinetic_miniapp/`:

```bash
module load clang/20.1.8
# Format the ckm C++ you touched (tracked and new files, excluding submodules):
git ls-files -co --exclude-standard '*.cpp' '*.hpp' | grep -v '^external/' \
  | xargs -r clang-format -i
# Confirm clean:
git ls-files -co --exclude-standard '*.cpp' '*.hpp' | grep -v '^external/' \
  | xargs -r clang-format --dry-run --Werror
script/check_boundary.sh
```

## Notes

- **Python.** If you changed any `.py` file, also run `black` on it. CI uses `black`.
- **Copyright headers.** `../script/format.sh` also adds riot's copyright header to
  files that lack one. New ckm files should already have it (see `AGENTS.md`).
- **Prefer the targeted commands above** to running `format.sh` itself. It touches the
  whole repo and needs `black` on PATH.

Report whether formatting changed any files (list them), and the boundary-check
result.
