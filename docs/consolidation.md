# Consolidation record

Date: 2026-09-28. Destination: `/Users/cyan/Desktop/tmp/crypto-research`.

## Source mapping

| Source | Destination |
| --- | --- |
| Local `/Users/cyan/Desktop/tmp/crypto` legacy source, package files and scripts | `projects/base-grid/` |
| Local `experiments/hl20/` | `experiments/hl20/` |
| Local `docs/` | `docs/audits/` |
| Local `scripts/audit_hyperliquid_remote.py` | `scripts/audit_hyperliquid_remote.py` |
| Local `.sisyphus/` | `archive/local-agent-notes/` (untracked) |
| SSH U `/home/cyan/default/crypto`, excluding research and generated environments | `projects/crypto-transformer/` |
| SSH U `research/hl20_20260924/` | `archive/server-research/hl20_20260924/` (untracked) |

Original source folders remain in place. Existing source, reports and frozen experiment files are copied without changing their contents. Historical absolute paths in reports refer to the original locations. Run each historical project from its own directory.

## Storage policy

- `.env` files are left only in the original source folders. No private keys or account configuration are intentionally imported.
- Generated dependencies (`node_modules`, `.venv`), Python caches, package metadata and pytest caches are excluded from migration.
- Historical datasets, model checkpoints, logs and trading state are copied under each project's `data/` and ignored by Git. The server data directory is approximately 426 MiB.
- Server `tmp.json` is retained locally and ignored as an unidentified scratch artifact.
- The complete server data-collection snapshot and local agent notes are retained under ignored `archive/`. The Binance reference dataset in that snapshot is not treated as Hyperliquid execution data.
- Frozen Hyperliquid public market data, manifests, result files and report images are versioned for reproducibility. Raw public API responses are restored from the server snapshot into the experiment's `data/raw/` directory when the normalized dataset matches.
- No trading processes or credentials are activated by this migration.

## Validation

The migrated Hyperliquid suite passed all 54 tests. Further copy-integrity and repository verification evidence is recorded in `migration-verification.json` alongside this document.

The Base project passed 174 tests and TypeScript checking; the original server Python project passed 15 tests. Its copied source was compared with the server by SHA-256. npm reported nine existing dependency vulnerabilities (three moderate, five high, one critical); dependencies were not upgraded during consolidation.

Git whitespace checking passed for the imported project source and new documentation. The preserved, generated `selection_comparison.svg` contains 5,146 trailing-whitespace findings; its original bytes were retained to preserve the experiment artifact.
