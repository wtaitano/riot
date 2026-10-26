# Plan: make the mattpocock-skills plugin reach every teammate (wtaitano/riot#5, stacked on #3)

Generated with Claude Code (agentic AI).

## Problem

The shared `.claude/settings.json` enabled `mattpocock-skills@claude-plugins-official`,
and the docs said it was "installed on first trusted launch". It is not. In the
official marketplace the plugin's entry points at an external source
(`github.com/mattpocock/skills`), and Claude Code does not fetch an externally sourced
plugin whose only `true` is in project settings. Each teammate instead sees
`Plugin "mattpocock-skills" is enabled in project settings but isn't installed here`
until they run `claude plugin install ... --scope project` by hand. It worked for the
maintainer only because they had installed it manually.

## Decision

Register Matt Pocock's own marketplace in the shared settings and enable the plugin
from it:

```json
"extraKnownMarketplaces": {
  "mattpocock": {
    "source": { "source": "github", "repo": "mattpocock/skills", "ref": "v1.3.1" }
  }
},
"enabledPlugins": { "mattpocock-skills@mattpocock": true }
```

- **No install step.** In that marketplace the plugin is the repository itself (a
  relative-path source, `./`). Per the Claude Code plugin docs, a relative-path plugin
  loads from the marketplace copy once the repository's `extraKnownMarketplaces`
  entry applies, which happens after the workspace-trust prompt that teammates
  already accept for the hooks.
- **Pinned.** `ref` is the release tag `v1.3.1` (the version the official entry
  currently resolves to). Third-party marketplaces do not auto-update by default, so
  upgrades happen only when someone changes the tag here. This replaces the earlier
  "unpinned, drift accepted" choice in `ckm-dev-env.md`.
- **Network.** The clone is from github.com, which the sandbox already allows.
- **Same code.** The official entry pointed at the same repository; only the
  catalog that names it changes.

Rejected: keep the official entry and add a per-teammate `claude plugin install`
line to the README. It works, but adds a manual step and leaves the version floating.

## Changes

- `.claude/settings.json`: the block above, replacing the old `enabledPlugins` entry.
- `README.md` "Getting started" step 6: no plugin prompt; check `/plugin`.
- `doc/sphinx/src/contributing.rst`: the shared-config table row.

## Testing

- `settings.json` parses as JSON; the boundary check passes; the docs build.
- **Live check (human):** in a fresh clone, launch `claude` from the miniapp, accept
  trust, and confirm `/plugin` lists `mattpocock-skills@mattpocock` with no Errors-tab
  entry and that `/mattpocock-skills:tdd` is offered.
