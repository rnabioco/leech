**A channel-ablation workflow for the Snakemake pipeline (`use_channel_ablation: true`).**
Building on #358's feature-channels-as-data work, `pipeline/workflow/rules/ablation.smk`
sweeps `leech model train --feature-channels` across configurable arms --
`baseline` (the default twelve), leave-one-out per channel, add-one for
`level_skew`/`level_kurtosis`, and free-form `custom` lists -- each repeated
over several training seeds so a channel's effect can be judged against
seed-to-seed noise. The train/val/test corpus is prepared and merged *once*
per pairwise comparison with `--feature-channels all`, under a path that
names that corpus, and reused by every arm; only the training call varies.
The aggregate rule writes one compressed table
(`channel_ablation_comparison.tsv.gz`) with a row per (pair, arm, seed),
`leech eval test`'s own metrics, and, per (pair, arm), the mean/sd across
seeds and the delta against that pair's baseline. Off by default, and not
part of `rule all`. `pipeline/config/config.yaml` also gains a top-level
`feature_channels` key threaded through `prepare_chunks` and the grid-search
re-prepare rules as `--feature-channels`, independent of the ablation
workflow. See [Channel ablation](docs/pipeline.md#channel-ablation).
