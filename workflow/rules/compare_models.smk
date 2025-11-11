"""
Model comparison rules for evaluating multiple architectures.

These rules train and evaluate multiple model architectures on the same
data splits to enable fair comparison.
"""

# ============================================================================
# Multi-Architecture Training: Charged vs Uncharged
# ============================================================================


rule train_architecture_charged:
    """Train a specific architecture for charged vs uncharged classification."""
    input:
        train=CHUNKS_DIR + "/merged/charged_vs_uncharged/train.npz",
        val=CHUNKS_DIR + "/merged/charged_vs_uncharged/val.npz",
        grid_search=(
            ancient(
                MODELS_DIR
                + "/grid_search/charged_vs_uncharged/{architecture}/best_params.json"
            )
            if config.get("use_grid_search", False)
            else []
        ),
    output:
        model=MODELS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/model_best.pt",
        checkpoint=MODELS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/model_checkpoint.pt",
        history=MODELS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/training_history.json",
    params:
        output_dir=MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}",
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        config_flag=lambda wildcards, input: (
            f"--config {input.grid_search}"
            if config.get("use_grid_search", False)
            else ""
        ),
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=240,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=16000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/train.log",
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule test_architecture_charged:
    """Test a specific architecture on the merged test set."""
    input:
        model=MODELS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/model_best.pt",
        test=CHUNKS_DIR + "/merged/charged_vs_uncharged/test.npz",
    output:
        metrics=METRICS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/test_metrics.json",
    log:
        METRICS_DIR
        + "/comparison/charged_vs_uncharged/{architecture}/test.log",
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            2>&1 | tee {log}
        """


rule compare_architectures_charged:
    """Compare all architectures on charged vs uncharged task."""
    input:
        expand(
            METRICS_DIR
            + "/comparison/charged_vs_uncharged/{arch}/test_metrics.json",
            arch=MODEL_ARCHITECTURES,
        ),
    output:
        comparison=METRICS_DIR
        + "/comparison/charged_vs_uncharged_architecture_comparison.tsv",
        summary=METRICS_DIR
        + "/comparison/charged_vs_uncharged_architecture_summary.txt",
    params:
        metrics_dir=METRICS_DIR + "/comparison/charged_vs_uncharged",
        architectures=MODEL_ARCHITECTURES,
    script:
        "../scripts/compare_architectures.py"


# ============================================================================
# Grid Search per Architecture (optional)
# ============================================================================


rule grid_search_architecture_charged:
    """Perform grid search for a specific architecture on charged vs uncharged."""
    input:
        train=CHUNKS_DIR + "/merged/charged_vs_uncharged/train.npz",
        val=CHUNKS_DIR + "/merged/charged_vs_uncharged/val.npz",
    output:
        results=MODELS_DIR
        + "/grid_search/charged_vs_uncharged/{architecture}/grid_search_results.json",
        best_params=MODELS_DIR
        + "/grid_search/charged_vs_uncharged/{architecture}/best_params.json",
    params:
        output_dir=MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}",
        param_grid=config.get(
            "grid_search",
            {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [64, 128, 256],
                "hidden_size": [128, 256, 512],
                "num_layers": [1, 2, 3],
            },
        ),
        max_epochs=config.get("grid_search_epochs", 20),
    log:
        MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}/grid_search.log",
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """


# ============================================================================
# Multi-Architecture Training: Pairwise Amino Acid Comparisons
# ============================================================================


rule train_architecture_pairwise:
    """Train a specific architecture for pairwise amino acid classification."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
        grid_search=(
            ancient(
                MODELS_DIR
                + "/grid_search/pairwise/{pair}/{architecture}/best_params.json"
            )
            if config.get("use_grid_search", False)
            else []
        ),
    output:
        model=MODELS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        checkpoint=MODELS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/model_checkpoint.pt",
        history=MODELS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/training_history.json",
    params:
        output_dir=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}",
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        config_flag=lambda wildcards, input: (
            f"--config {input.grid_search}"
            if config.get("use_grid_search", False)
            else ""
        ),
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=240,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=16000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/train.log",
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule test_architecture_pairwise:
    """Test a specific architecture on pairwise test set."""
    input:
        model=MODELS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        metrics=METRICS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/test_metrics.json",
    params:
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "atesting_a100",
        runtime=lambda wildcards, attempt: 240 if config.get("use_cpu_training", False) else 60,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 2,
        mem_mb=4000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        METRICS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/test.log",
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule compare_architectures_pairwise:
    """Compare all architectures on a specific pairwise task."""
    input:
        expand(
            METRICS_DIR
            + "/comparison/pairwise/{{pair}}/{arch}/test_metrics.json",
            arch=MODEL_ARCHITECTURES,
        ),
    output:
        comparison=METRICS_DIR
        + "/comparison/pairwise/{pair}/architecture_comparison.tsv",
        summary=METRICS_DIR
        + "/comparison/pairwise/{pair}/architecture_summary.txt",
    params:
        metrics_dir=METRICS_DIR + "/comparison/pairwise/{pair}",
        architectures=MODEL_ARCHITECTURES,
    script:
        "../scripts/compare_architectures.py"


rule aggregate_pairwise_comparisons:
    """Aggregate architecture comparisons across all pairwise tasks."""
    input:
        expand(
            METRICS_DIR + "/comparison/pairwise/{pair}/architecture_comparison.tsv",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
    output:
        comparison=METRICS_DIR + "/comparison/pairwise_architecture_comparison.tsv",
        summary=METRICS_DIR + "/comparison/pairwise_architecture_summary.txt",
    run:
        import pandas as pd
        from pathlib import Path

        if not AA_PAIRS:
            # No pairwise comparisons, create empty files
            Path(output.comparison).touch()
            Path(output.summary).write_text("No pairwise comparisons configured.\n")
        else:
            # Read and concatenate all pairwise comparison results
            dfs = []
            for pair_file in input:
                pair_name = Path(pair_file).parent.name
                df = pd.read_csv(pair_file, sep="\t")
                df.insert(0, "pair", pair_name)
                dfs.append(df)

            # Concatenate and save
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(output.comparison, sep="\t", index=False)

            # Create summary
            with open(output.summary, "w") as f:
                f.write("Architecture Comparison Summary Across All Pairwise Tasks\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Total pairs evaluated: {len(AA_PAIRS)}\n")
                f.write(f"Architectures compared: {', '.join(MODEL_ARCHITECTURES)}\n\n")

                # Best architecture per pair
                f.write("Best Architecture by Pair (by accuracy):\n")
                f.write("-" * 70 + "\n")
                for pair in AA_PAIRS:
                    pair_df = combined[combined["pair"] == pair]
                    if not pair_df.empty:
                        best = pair_df.loc[pair_df["accuracy"].idxmax()]
                        f.write(f"{pair:20s} {best['architecture']:20s} acc={best['accuracy']:.4f}\n")

                f.write("\n")

                # Overall best architecture (average across pairs)
                f.write("Overall Best Architecture (average accuracy):\n")
                f.write("-" * 70 + "\n")
                avg_by_arch = combined.groupby("architecture")["accuracy"].mean().sort_values(ascending=False)
                for arch, acc in avg_by_arch.items():
                    f.write(f"{arch:20s} avg_acc={acc:.4f}\n")


rule grid_search_architecture_pairwise:
    """Perform grid search for a specific architecture on pairwise comparison."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
    output:
        results=MODELS_DIR
        + "/grid_search/pairwise/{pair}/{architecture}/grid_search_results.json",
        best_params=MODELS_DIR
        + "/grid_search/pairwise/{pair}/{architecture}/best_params.json",
    params:
        output_dir=MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}",
        param_grid=config.get(
            "grid_search",
            {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [64, 128, 256],
                "hidden_size": [128, 256, 512],
                "num_layers": [1, 2, 3],
            },
        ),
        max_epochs=config.get("grid_search_epochs", 20),
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=lambda wildcards, attempt: 1400 if config.get("use_cpu_training", False) else 480,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=16000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/grid_search.log",
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """
