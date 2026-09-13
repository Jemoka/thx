# Design

Architecture and design documents for theseus internals.

## Documents

- [Experiment DAG and Storage](dag.md) — node identities, ticking, branches, Parquet records, and ClickHouse queries.
- [Available Chips](chips.md) — all supported chip string keys, memory, and theoretical compute peaks.
- [Sharding Design](sharding.md) — execution policy, logical tensor axes, and parameter layouts.
- [Config System](config.md) — how `field()`, `build()`, and `configuration()` turn annotated dataclasses into a unified OmegaConf schema.
- [Analysis and Plotting](plot.md) — checkpoint inspection, trainer-backed analyses, and figure artifacts logged to Comet.
- [Dataloader I/O](dataloader-io.md) — latency-aware file reads, replay invariants, and measured storage behavior.
