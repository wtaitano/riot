<!-- This file was made in part with generative AI. -->

# Project map

The `project-map` agent (`.claude/agents/project-map.md`) draws a one-file HTML map of
ckm in the git-ignored `.project-map/`. Open `.project-map/index.html` to see it.

- Before working on your own for a long stretch, start `project-map` in the background
  to draw the current map. Pass it the milestones, decisions and test results you
  already know from the conversation.
- The first time (no `.project-map/state.json` yet), run it in the foreground instead.
  If it replies `STYLE_NEEDED`, ask the user "dark or light, and one accent color?" and
  run it again with the answer. If it drafted milestones, show them to the user.
- After each milestone is finished, have `project-map` update the map.
- When the user asks "where are we?", answer from `.project-map/state.json` and the
  map. If the map is older than the latest commit, refresh it first.
- Follow the map's suggested next step, unless the user has said to hold or asked for
  something else.
- Anything that needs the user's call goes in the map's "Needs your call" list, with a
  default. If the user hasn't answered by the time you reach it, go ahead with the
  default and say so.
- This does not cover irreversible or outward-facing actions (push, PRs, deleting,
  messaging others). Those still go to the user, through `scratch/commands.txt`.
