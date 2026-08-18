# Fork sync workflow

This repo is a fork of `lyonzin/knowledge-rag` with local patches for the
Xata repos (Go/YAML/SQL/Proto parsing, Xata config, word-boundary chunking).

## Branch layout

- `master` — tracks `upstream/master` (clean, no local changes)
- `xata` — upstream + local patches. **This is the working branch.**
- `upstream-parsers` — the parser additions, submitted as a PR to upstream
  (https://github.com/lyonzin/knowledge-rag/pull/194)

## Local patch series (on `xata`)

Kept small and focused so rebases stay conflict-free:

1. `feat(ingestion)` — expanded file format support (Go/YAML/Proto/SQL/Shell/etc.)
2. `fix(ingestion)` — restore C/C++/XML parsers + JS/TS extraction
3. `chore(config)` — Xata repo config (`config.yaml`, force-added)
4. `indexer` — word-boundary chunking + fuzzy near-duplicate dedup
5. `test` / `fix(dedup)` — test + dedup adaptations

## Sync with upstream

```bash
git fetch upstream
git rebase upstream/master   # on the xata branch
```

If a conflict appears, resolve it, `git add` the file, then `git rebase --continue`.
Because the patches are small and focused, conflicts are rare.

## After syncing

```bash
git push --force-with-lease origin xata
```

## Notes

- `config.yaml` is force-added to `xata` (upstream gitignores it). It is the
  Xata-specific config; `documents/` symlinks stay local and are gitignored.
- The old local work (pre-rebase) is preserved in `backup/before-rebase-v4`.
- The parser additions live in `upstream-parsers` for upstream contribution;
  once merged upstream, they can be dropped from the local patch series.
