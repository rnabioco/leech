"""
Rules for tRNA optimization model comparison.

These rules train and compare 4 model variants optimized for constant-sequence
tRNA aminoacylation:
  1. ConvLSTMBase: Signal + sequence only (no features) - baseline
  2. ConvLSTMDwell: Full model with all branches - standard
  3. TCNSignalFeatures: Signal + features only (no sequence) - optimized for tRNA
  4. ConvLSTMDwell with masking: Full model with 50% sequence masking

The comparison quantifies the impact of sequence overfitting on constant CCAGGC motif.
"""

# Model variants for tRNA optimization comparison
TRNA_VARIANTS = config.get(
    "tRNA_variants",
    ["base", "dwell", "tcn_signal_features", "dwell_masked"],
)

# Map variant names to model architectures
VARIANT_TO_MODEL = {
    "base": "ConvLSTMBase",
    "dwell": "ConvLSTMDwell",
    "tcn_signal_features": "TCNSignalFeatures",
    "dwell_masked": "ConvLSTMDwell",
}

# Map variant names to config files
VARIANT_TO_CONFIG = {
    "base": "configs/comparison/base.yaml",
    "dwell": "configs/comparison/dwell.yaml",
    "tcn_signal_features": "configs/comparison/tcn_signal_features.yaml",
    "dwell_masked": "configs/comparison/dwell_masked.yaml",
}


# ============================================================================
# Training Rules: tRNA Optimization Variants
# ============================================================================


rule train_tRNA_variant_pairwise:
    """Train a tRNA optimization variant for pairwise comparison."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
        config_file=lambda wildcards: VARIANT_TO_CONFIG.get(wildcards.variant, ""),
    output:
        model=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}/model_best.pt",
        checkpoint=MODELS_DIR
        + "/tRNA_optimization/pairwise/{pair}/{variant}/model_last.pt",
        history=MODELS_DIR
        + "/tRNA_optimization/pairwise/{pair}/{variant}/metrics.json",
    params:
        output_dir=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}",
        model_name=lambda wildcards: VARIANT_TO_MODEL[wildcards.variant],
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        # Add --mask-sequence-prob for dwell_masked variant
        mask_prob=lambda wildcards: (
            "--mask-sequence-prob 0.5" if wildcards.variant == "dwell_masked" else ""
        ),
        # Use config file if available
        config_flag=lambda wildcards, input: (
            f"--model-config {input.config_file}" if input.config_file else ""
        ),
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=480,
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 4
        ),
        mem_mb=16000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}/train.log",
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_name} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            {params.mask_prob} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule test_tRNA_variant_pairwise:
    """Test a tRNA optimization variant on pairwise test set."""
    input:
        model=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        metrics=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/{variant}/test_metrics.json",
    params:
        model_dir=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "atesting_a100"
        ),
        runtime=lambda wildcards, attempt: (
            240 if config.get("use_cpu_training", False) else 60
        ),
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 2
        ),
        mem_mb=4000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/{variant}/test.log",
    shell:
        """
        uv run leech test \
            --model {params.model_dir} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


# ============================================================================
# Analysis Rules: Model Comparison
# ============================================================================


rule compare_tRNA_variants_pairwise:
    """Compare all tRNA optimization variants using leech analyze compare."""
    input:
        models=expand(
            MODELS_DIR
            + "/tRNA_optimization/pairwise/{{pair}}/{variant}/model_best.pt",
            variant=TRNA_VARIANTS,
        ),
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        comparison=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/comparison_results.json",
        summary=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/comparison_summary.txt",
        curves=METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/training_curves.png",
    params:
        model_dirs=lambda wildcards, input: " ".join(
            [
                f"-m {MODELS_DIR}/tRNA_optimization/pairwise/{wildcards.pair}/{v}"
                for v in TRNA_VARIANTS
            ]
        ),
        output_dir=METRICS_DIR + "/tRNA_optimization/pairwise/{pair}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "atesting_a100"
        ),
        runtime=120,
        cpus_per_task=4,
        mem_mb=8000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/compare.log",
    shell:
        """
        uv run leech analyze compare \
            {params.model_dirs} \
            -t {input.test} \
            -o {params.output_dir} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule feature_importance_tRNA_pairwise:
    """Compute feature importance for best tRNA variant."""
    input:
        model=MODELS_DIR
        + "/tRNA_optimization/pairwise/{pair}/tcn_signal_features/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        importance=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/feature_importance/importance_scores.npz",
        plot=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/feature_importance/importance_plot.png",
    params:
        model_dir=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/tcn_signal_features",
        output_dir=METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/feature_importance",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "atesting_a100"
        ),
        runtime=120,
        cpus_per_task=4,
        mem_mb=8000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/feature_importance/feature_importance.log",
    shell:
        """
        uv run leech analyze feature-importance \
            -m {params.model_dir} \
            -t {input.test} \
            -o {params.output_dir} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule sequence_ablation_tRNA_pairwise:
    """Test sequence contribution for full model (dwell) variant."""
    input:
        model=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/dwell/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        ablation=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_results.json",
        plot=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_plot.png",
    params:
        model_dir=MODELS_DIR + "/tRNA_optimization/pairwise/{pair}/dwell",
        output_dir=METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/sequence_ablation",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "atesting_a100"
        ),
        runtime=120,
        cpus_per_task=4,
        mem_mb=8000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/sequence_ablation/sequence_ablation.log",
    shell:
        """
        uv run leech analyze sequence-ablation \
            -m {params.model_dir} \
            -t {input.test} \
            -o {params.output_dir} \
            --device {params.device} \
            2>&1 | tee {log}
        """


# ============================================================================
# Aggregate Rules: Combine Results Across Pairs
# ============================================================================


rule aggregate_tRNA_comparison:
    """Aggregate tRNA variant comparisons across all pairwise tasks."""
    input:
        expand(
            METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/comparison_results.json",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
    output:
        aggregate=METRICS_DIR + "/tRNA_optimization/aggregate/variant_comparison.tsv",
        summary=METRICS_DIR + "/tRNA_optimization/aggregate/variant_summary.txt",
    run:
        import json
        import pandas as pd
        from pathlib import Path

        if not AA_PAIRS:
            Path(output.aggregate).touch()
            Path(output.summary).write_text("No pairwise comparisons configured.\n")
        else:
            # Read and parse all comparison results
            rows = []
            for pair_file in input:
                pair_name = Path(pair_file).parent.name
                with open(pair_file) as f:
                    comparison = json.load(f)

                for variant, metrics in comparison.items():
                    row = {"pair": pair_name, "variant": variant}
                    row.update(metrics)
                    rows.append(row)

                    # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output.aggregate, sep="\t", index=False)

            # Create summary
            with open(output.summary, "w") as f:
                f.write("tRNA Optimization Variant Comparison Summary\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total pairs evaluated: {len(AA_PAIRS)}\n")
                f.write(f"Variants compared: {', '.join(TRNA_VARIANTS)}\n\n")

                # Best variant per pair
                f.write("Best Variant by Pair (by accuracy):\n")
                f.write("-" * 80 + "\n")
                for pair in AA_PAIRS:
                    pair_df = df[df["pair"] == pair]
                    if not pair_df.empty:
                        best = pair_df.loc[pair_df["accuracy"].idxmax()]
                        f.write(
                            f"{pair:20s} {best['variant']:25s} acc={best['accuracy']:.4f}\n"
                        )

                f.write("\n")

                # Overall best variant (average across pairs)
                f.write("Overall Best Variant (average accuracy):\n")
                f.write("-" * 80 + "\n")
                avg_by_variant = (
                    df.groupby("variant")["accuracy"]
                    .mean()
                    .sort_values(ascending=False)
                )
                for variant, acc in avg_by_variant.items():
                    f.write(f"{variant:25s} avg_acc={acc:.4f}\n")

                f.write("\n")

                # Performance improvement vs baseline
                f.write("Performance vs Baseline (ConvLSTMBase):\n")
                f.write("-" * 80 + "\n")
                baseline_acc = df[df["variant"] == "base"]["accuracy"].mean()
                for variant in TRNA_VARIANTS:
                    if variant == "base":
                        continue
                    variant_acc = df[df["variant"] == variant]["accuracy"].mean()
                    improvement = ((variant_acc - baseline_acc) / baseline_acc) * 100
                    f.write(
                        f"{variant:25s} {improvement:+6.2f}% improvement over baseline\n"
                    )


# ============================================================================
# Convenience Rules
# ============================================================================


rule tRNA_optimization_full_analysis:
    """Run complete tRNA optimization analysis for a specific pair."""
    input:
        comparison=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/comparison_results.json",
        feature_importance=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/feature_importance/importance_scores.npz",
        sequence_ablation=METRICS_DIR
        + "/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_results.json",


rule tRNA_optimization_all:
    """Run complete tRNA optimization analysis for all configured pairs."""
    input:
        expand(
            METRICS_DIR + "/tRNA_optimization/pairwise/{pair}/comparison_results.json",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
        expand(
            METRICS_DIR
            + "/tRNA_optimization/pairwise/{pair}/feature_importance/importance_scores.npz",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
        expand(
            METRICS_DIR
            + "/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_results.json",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
        METRICS_DIR + "/tRNA_optimization/aggregate/variant_comparison.tsv",
