---
name: ckm-format
description: Format ckm C++ with riot's clang-format 20 config and check it is clean; also check the isolation boundary. Use before committing any ckm change.
---

<!-- This file was made in part with generative AI. -->

Riot's CI (`format.yml`) runs `../script/format.sh` over every tracked `.cpp`, `.hpp`
and `.py` file, including ckm's. Do the same checks locally. Run from
`conditional_kinetic_miniapp/`:

```bash
module load clang/20.1.8
# Format the ckm C++ files, tracked and new:
git ls-files -co --exclude-standard '*.cpp' '*.hpp' | xargs -r clang-format -i
# Confirm they are clean:
git ls-files -co --exclude-standard '*.cpp' '*.hpp' | xargs -r clang-format --dry-run --Werror
script/check_boundary.sh
```

## Notes

- **Python files.** If you changed any `.py` file, also run `black` on it, because CI
  uses `black`.
- **Riot files.** If you edited a riot file (only the allowed wiring files), format it
  with riot's tools as well.
- **Copyright headers.** `../script/format.sh` adds riot's copyright header to files
  that lack it. New ckm files should already have one; see `AGENTS.md`.

## Report

Say whether formatting changed any files and list them, then give the boundary-check
result.
