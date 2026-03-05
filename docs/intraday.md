# Intraday Runbook

The intraday lab extends the existing `src/lab/` MVP into a paper-only day-trading workflow.

## Commands

Run research:

```bash
./tool research --last-years 3 --timeframes H1,M15
```

Replay the latest champion in paper mode:

```bash
./tool paper-run --champion reports/champion.json --mode replay --days 5
```

Run paper-live polling (still simulated fills, still paper only):

```bash
./tool paper-run --champion reports/champion.json --mode paper-live --days 1
```

## What The Research Pipeline Does

1. Loads cached bars from `data/cache/` via `BarStore`, fetching from Yahoo Finance only when cache is missing.
2. Evaluates enabled strategies on configured timeframes.
3. Uses purged, embargoed walk-forward validation.
4. Enforces explicit frequency guardrails per timeframe.
5. Ranks candidates by out-of-sample score, stability, frequency, drawdown, baseline margin, and parameter robustness.
6. Writes reports to `reports/<run>/` and a versioned champion artifact to `artifacts/champions/`.
7. Runs a deterministic paper replay from the chosen champion.

## Safety Guardrails

- Live trading is hard-disabled in config validation.
- `LiveBrokerAdapter` raises unless future work explicitly enables both config and env allowlist flags.
- End-of-day flatten is on by default.
- Overnight holds are off by default.
- Daily loss, max trades/day, max notional, and consecutive-loss lockouts are enforced before entries.

## Provider Limits

Yahoo Finance intraday coverage is best-effort:

- `H1`: up to about 730 days
- `M15` / `M5`: up to about 60 days

When coverage is truncated, the report and champion artifact include explicit coverage metadata and warnings.
