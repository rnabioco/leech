"""
Channel-ablation workflow: does dropping or adding a feature channel change
the model's held-out performance?

Opt-in (`use_channel_ablation: true` in config.yaml), and only included by
the Snakefile when that switch is set. Mirrors grid_search.smk's and
compare_models.smk's grid-search-optimized re-prepare/train/test/compare
shape: a re-prepare + merge handoff, a per-arm training+eval job, and an
aggregate rule -- reusing `common.smk`'s helpers, the same output-directory
nesting under CHUNKS_DIR/MODELS_DIR/METRICS_DIR, and the same
Slurm-profile-owns-resources convention (see pipeline/cluster/slurm{,-cpu}
/config.yaml's `ablation_*` entries).

The load-bearing property, called out in the config comment and repeated
here: the train/val/test corpus is prepared and merged ONCE per pair, with
`--feature-channels all` (every known channel -- the default twelve plus
level_skew/level_kurtosis), under a path that names that corpus
("ablation/all_channels") rather than any one arm. Every arm/seed then reads
the SAME merged files and selects its own channel subset at `model train
--feature-channels` time, from that corpus's recorded columns. A corpus per
arm would be wasteful, byte-identical work repeated `len(arms)` times over
data that is already large (see ~/.claude/CLAUDE.md's rule against
duplicating large derived corpora).
"""

from leech.constants import DEFAULT_FEATURE_CHANNELS


def _build_ablation_arms():
    """Resolve the `ablation:` config block into {arm_name: [channels]}.

    Always includes "baseline" (the default twelve). Leave-one-out and
    add-one arms are generated from config lists; `custom` arms are taken
    verbatim. Channel names are not validated against
    `leech.constants.KNOWN_FEATURE_CHANNELS` here -- `leech model train`
    already refuses an unknown name before touching any data, and duplicating
    that check would just be a second place for the known-channel set to go
    stale.
    """
    ablation_cfg = config.get("ablation") or {}
    baseline = list(DEFAULT_FEATURE_CHANNELS)
    arms = {"baseline": baseline}

    leave_one_out = ablation_cfg.get("leave_one_out", DEFAULT_FEATURE_CHANNELS)
    for channel in leave_one_out:
        if channel not in baseline:
            raise WorkflowError(
                f"ablation.leave_one_out channel '{channel}' is not in the "
                f"default twelve ({baseline}); leave-one-out only makes "
                "sense over channels the baseline arm actually has."
            )
        arms[f"loo_{channel}"] = [c for c in baseline if c != channel]

    add_one = ablation_cfg.get(
        "add_one",
        [["level_skew"], ["level_kurtosis"], ["level_skew", "level_kurtosis"]],
    )
    for extra in add_one:
        extra = list(extra)
        name = "add_" + "_".join(extra)
        if name in arms:
            raise WorkflowError(
                f"ablation.add_one produced a duplicate arm name '{name}'"
            )
        arms[name] = baseline + extra

    custom = ablation_cfg.get("custom") or {}
    for name, channels in custom.items():
        if name in arms:
            raise WorkflowError(
                f"ablation.custom arm name '{name}' collides with a "
                "built-in baseline/leave-one-out/add-one arm name"
            )
        arms[name] = list(channels)

    return arms


ABLATION_ARMS = _build_ablation_arms()
ABLATION_ARM_NAMES = list(ABLATION_ARMS.keys())
ABLATION_SEEDS = (config.get("ablation") or {}).get("seeds", [1, 2, 3])
ABLATION_BASELINE_ARM = "baseline"


rule ablation_prepare_chunks:
    """Re-extract one sample's chunks with every known feature channel.

    The channel-ablation twin of `reprepare_chunks_optimized_pairwise`
    (grid_search.smk): same handoff shape (re-run `data prepare` from the
    original pod5/bam, this time under an `{pair}`-scoped output path), but
    the axis that changes is `--feature-channels`, not `--signal-context`,
    and the value is always the literal "all" -- never the global
    `feature_channels` config key (`build_feature_channels_arg()`), since
    every arm needs every channel available to select from at train time.
    """
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
    output:
        all=CHUNKS_DIR + "/ablation/all_channels/{pair}/{sample}/all.npz",
    log:
        CHUNKS_DIR + "/ablation/all_channels/{pair}/{sample}/prepare.log",
    # sample/pair are constrained globally (Snakefile).
    threads: config.get("workers", 4)
    params:
        output_dir=CHUNKS_DIR + "/ablation/all_channels/{pair}/{sample}",
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        motif_reference=config.get("motif_reference", "fasta"),
        workers=config.get("workers", 4),
        chunk_size=config.get("chunk_size", 100),
        ref_fasta_arg=build_reference_fasta_arg(),
        skip_indels_arg=build_skip_indels_arg(),
        label_arg=lambda wildcards: build_label_arg(wildcards),
        signal_context_arg=build_signal_context_arg(),
    shell:
        """
        uv run leech data prepare \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --motif-reference {params.motif_reference} \
            {params.ref_fasta_arg} \
            {params.skip_indels_arg} \
            {params.label_arg} \
            {params.signal_context_arg} \
            --feature-channels all \
            --workers {params.workers} \
            --chunk-size {params.chunk_size} \
            --no-split \
            2>&1 | tee {log}
        """


rule ablation_merge_chunks:
    """Merge the all-channel ablation chunks and split at read level.

    Mirrors `merge_chunks_pairwise`/`merge_chunks_optimized_pairwise` exactly,
    over `ablation_prepare_chunks`'s output. Runs ONCE per pair -- every arm
    and seed trains against these same three files.
    """
    input:
        chunks=lambda wildcards: expand(
            CHUNKS_DIR + "/ablation/all_channels/{{pair}}/{sample}/all.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
    output:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/val.npz",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/test.npz",
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/merge_and_split.log",
    params:
        output_dir=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        input_args=lambda wildcards, input: build_merge_input_args(
            wildcards.pair, input.chunks
        ),
    shell:
        """
        uv run leech data merge \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            2>&1 | tee {log}
        """


rule ablation_train:
    """Train one (arm, seed) on the shared all-channel corpus.

    `--feature-channels` selects and orders the subset of the corpus's
    recorded columns this arm trains on; `--seed` is the only other thing
    that varies within an arm, to measure run-to-run noise against the
    channel effect. Model architecture is the single-model `model` config
    key (this workflow ablates channels for one architecture, not
    architectures -- see compare_models.smk for that axis).
    """
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/val.npz",
    output:
        model=MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}/model_best.pt",
        checkpoint=MODELS_DIR
        + "/ablation/pairwise/{pair}/{arm}/seed{seed}/model_last.pt",
        history=MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}/metrics.json",
    log:
        MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}/train.log",
    # slurm_partition/runtime/cpus_per_task/mem_mb/gres for this rule live in
    # the cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) -- see
    # train.smk's train_pairwise_aa for why. Always single-GPU (no
    # `train_gpus` lambda needed), like train_architecture_pairwise.
    params:
        output_dir=MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}",
        model_type=config.get("model", "ConvLSTMDwell"),
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        channels=lambda wildcards: ",".join(ABLATION_ARMS[wildcards.arm]),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    # --resume points at the undeclared rolling checkpoint (model_resume.pt),
    # not either declared output -- see train_pairwise_aa in train.smk and
    # leech#330 for why. Placed above shell: rather than between it and the
    # docstring, unlike train_pairwise_aa / train_architecture_pairwise: that
    # placement hides every flag in the block from the static flag check in
    # tests/test_pipeline_rules.py, a pre-existing gap this rule does not
    # need to share.
    shell:
        """
        uv run leech model train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --feature-channels {params.channels} \
            --seed {wildcards.seed} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            --resume {params.output_dir}/model_resume.pt \
            2>&1 | tee {log}
        """


rule ablation_test:
    """Evaluate one (arm, seed) on the shared held-out test set.

    Same `leech eval test` invocation as test_pairwise_aa/
    test_architecture_pairwise -- the primary-metric convention
    (compare_architectures.py's f1-then-accuracy preference) is applied at
    aggregation time, not here.
    """
    input:
        model=MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/ablation/all_channels/test.npz",
    output:
        metrics=METRICS_DIR
        + "/ablation/pairwise/{pair}/{arm}/seed{seed}/test_metrics.json",
    log:
        METRICS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}/test.log",
    # Resources for this rule live in the cluster profile -- see
    # ablation_train above.
    params:
        model_dir=MODELS_DIR + "/ablation/pairwise/{pair}/{arm}/seed{seed}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    shell:
        """
        uv run leech eval test \
            --model {params.model_dir} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule ablation_aggregate:
    """Aggregate channel-ablation metrics across arms, seeds, and pairs.

    One compressed table (never plain multi-GiB text -- see
    ~/.claude/CLAUDE.md) with a row per (pair, arm, seed) plus, per (pair,
    arm), the mean/sd across seeds and the delta against that pair's
    baseline arm -- computed from `leech eval test`'s own metrics
    (accuracy/precision/recall/f1/auroc/auprc), nothing invented. See
    docs/pipeline.md#channel-ablation for how to read it: a delta smaller
    than that arm's seed-to-seed sd is not evidence.
    """
    input:
        expand(
            METRICS_DIR
            + "/ablation/pairwise/{pair}/{arm}/seed{seed}/test_metrics.json",
            pair=AA_PAIRS,
            arm=ABLATION_ARM_NAMES,
            seed=ABLATION_SEEDS,
        )
        if AA_PAIRS
        else [],
    output:
        comparison=METRICS_DIR
        + "/ablation/aggregate/channel_ablation_comparison.tsv.gz",
        summary=METRICS_DIR + "/ablation/aggregate/channel_ablation_summary.txt",
    params:
        metrics_dir=METRICS_DIR + "/ablation/pairwise",
        pairs=AA_PAIRS,
        arms=ABLATION_ARM_NAMES,
        seeds=ABLATION_SEEDS,
        arm_channels=ABLATION_ARMS,
        baseline_arm=ABLATION_BASELINE_ARM,
    script:
        "../scripts/aggregate_ablation.py"
