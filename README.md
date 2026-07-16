# Pixi Star History

A daily-updated comparison of GitHub stars for [`conda/conda`](https://github.com/conda/conda), [`prefix-dev/pixi`](https://github.com/prefix-dev/pixi), and [`mamba-org/mamba`](https://github.com/mamba-org/mamba).

**Live graph:** https://baszalmstra.github.io/pixi-star-history/

The page contains one interactive, light-themed chart with historical totals, observed archive anchors, six-month forecasts, empirical uncertainty bands, and detailed hover information.

## Why the history is estimated

Since July 2026, GitHub limits the stargazer-list endpoint to repository administrators and collaborators. The aggregate `stargazerCount` remains public, but GitHub does not provide historical aggregate counts or unstar events.

The initial history therefore combines three sources:

1. **Wayback Machine snapshots** of GitHub repository pages provide irregular historical aggregate counts. These anchors reflect net stars at the time, including earlier unstars.
2. **GH Archive WatchEvents**, queried through the public ClickHouse playground, provide the daily activity shape between anchors. Known old names are included for repositories that moved organizations.
3. **GitHub GraphQL** provides today's exact aggregate count and one exact snapshot on every subsequent daily run.

The resulting daily values between observed anchors are estimates, not exact measurements. The source classification is retained in [`data/star_history.csv`](data/star_history.csv), and every parsed Wayback observation and source URL is retained in [`data/wayback_anchors.csv`](data/wayback_anchors.csv).

### Observation types

| Value | Meaning |
|---|---|
| `wayback` | Aggregate count parsed from an archived GitHub page. |
| `snapshot` | Exact aggregate count queried directly from GitHub. |
| `estimated` | Interpolation between observations, shaped by archived WatchEvent activity. |

## Forecasting

Each repository gets a 183-day forecast from six candidate trend models:

- linear trends fitted over 30, 90, 180, and 365 days;
- damped trends fitted over 90 and 180 days.

The candidates are evaluated on several recent 30-day historical holdouts. Their weights are inverse to squared backtest error. Forecasts are constrained to be nondecreasing, while 95% ranges combine empirical backtest error and disagreement between candidate models. The graph exposes the strongest model weights and backtest RMSE in forecast tooltips.

Pairwise overtake dates come from 12,000 deterministic Monte Carlo paths per repository. Each path samples a candidate according to its backtested weight and adds temporally accumulated error calibrated to backtest RMSE. Repository errors are sampled independently. For every challenger, the graph reports:

- the probability of overtaking within six months;
- the median crossing date and star count;
- a conditional 95% crossing-date range among paths that cross;
- crossing-date standard deviation and variance.

Diamonds mark median overtake dates and pale vertical bands mark conditional 95% date ranges. Pairings below a 10% six-month crossing probability are omitted from the graph.

These projections extrapolate trends; they are exploratory rather than promises.

## Local development

Everything runs through [Pixi](https://pixi.sh/):

```shell
pixi install
pixi run check
pixi run build-site
```

Refresh today's exact aggregate snapshot and rebuild the graph:

```shell
pixi run refresh
```

Rebuild all estimated history, including Wayback and GH Archive queries:

```shell
pixi run collect --full
pixi run build-site
```

The full backfill makes many polite requests to the Internet Archive and should only be run when the historical source data needs to be reconstructed. The normal daily refresh makes one GitHub GraphQL request.

Authentication is resolved from `GITHUB_TOKEN`, `GH_TOKEN`, or the local GitHub CLI. No special stargazer-list permission or custom PAT is required because the project only requests public aggregate counts from GitHub.

## Data format

[`data/star_history.csv`](data/star_history.csv) is a long-form UTC panel:

```text
date,repository,stars,daily_change,observation
2026-07-16,conda/conda,7473,0,snapshot
```

There is one row per repository per recorded date, starting at the first archived star activity for that repository.

## Automation

[`.github/workflows/update-and-deploy.yml`](.github/workflows/update-and-deploy.yml) runs daily at **06:00 UTC** and can also be triggered manually. It:

1. takes exact GitHub aggregate snapshots;
2. runs linting and tests through Pixi;
3. regenerates and commits the CSV and HTML;
4. deploys `site/` with GitHub Pages.

## License

The code is available under the [MIT License](LICENSE). Derived source data retains the terms and limitations of GitHub, GH Archive, ClickHouse, and the Internet Archive.
