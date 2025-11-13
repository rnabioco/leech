# tRNA Optimization Snakemake Workflow

This document describes the Snakemake workflow for comparing model variants optimized for constant-sequence tRNA aminoacylation.

## Overview

The tRNA optimization workflow trains and compares 4 model variants designed to address sequence overfitting in constant-sequence applications:

1. **ConvLSTMBase** (`base`): Baseline with signal + sequence only (no features)
2. **ConvLSTMDwell** (`dwell`): Standard full model with all 3 branches
3. **TCNSignalFeatures** (`tcn_signal_features`): Optimized TCN with signal + features only (no sequence)
4. **ConvLSTMDwell with masking** (`dwell_masked`): Full model with 50% sequence masking

## Background

For tRNA aminoacylation, the sequence context is **constant** (CCAGGC motif) across all training examples. This causes the sequence branch to overfit to the constant pattern, wasting model capacity and reducing generalization.

The workflow quantifies the impact of this overfitting and validates the new signal+features architectures.

## Workflow Rules

### Training Rules

#### `train_tRNA_variant_pairwise`
Trains a specific variant for a pairwise comparison.

**Wildcards:**
- `{pair}`: Pairwise comparison (e.g., "charged_uncharged", "Ala_Gly")
- `{variant}`: Model variant (base, dwell, tcn_signal_features, dwell_masked)

**Output:**
- `results/models/tRNA_optimization/pairwise/{pair}/{variant}/model_best.pt`
- `results/models/tRNA_optimization/pairwise/{pair}/{variant}/model_last.pt`
- `results/models/tRNA_optimization/pairwise/{pair}/{variant}/training_history.json`

**Features:**
- Uses configuration files from `configs/comparison/{variant}.yaml`
- Automatically applies `--mask-sequence-prob 0.5` for `dwell_masked` variant
- Respects `use_cpu_training` config for CPU-only clusters

#### `test_tRNA_variant_pairwise`
Evaluates a trained variant on test data.

**Output:**
- `results/metrics/tRNA_optimization/pairwise/{pair}/{variant}/test_metrics.json`

### Analysis Rules

#### `compare_tRNA_variants_pairwise`
Compares all 4 variants using `leech analyze compare`.

**Output:**
- `results/metrics/tRNA_optimization/pairwise/{pair}/comparison_results.json`
- `results/metrics/tRNA_optimization/pairwise/{pair}/comparison_summary.txt`
- `results/metrics/tRNA_optimization/pairwise/{pair}/training_curves.png`

#### `feature_importance_tRNA_pairwise`
Computes gradient-based feature importance for TCNSignalFeatures (best variant).

**Output:**
- `results/metrics/tRNA_optimization/pairwise/{pair}/feature_importance/importance_scores.npz`
- `results/metrics/tRNA_optimization/pairwise/{pair}/feature_importance/importance_plot.png`

#### `sequence_ablation_tRNA_pairwise`
Tests sequence contribution by ablating sequence information for the full model.

**Output:**
- `results/metrics/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_results.json`
- `results/metrics/tRNA_optimization/pairwise/{pair}/sequence_ablation/ablation_plot.png`

### Aggregate Rules

#### `aggregate_tRNA_comparison`
Combines comparison results across all pairwise tasks.

**Output:**
- `results/metrics/tRNA_optimization/aggregate/variant_comparison.tsv`
- `results/metrics/tRNA_optimization/aggregate/variant_summary.txt`

**Summary includes:**
- Best variant per pair (by accuracy)
- Overall best variant (average across pairs)
- Performance improvement vs baseline

### Convenience Rules

#### `all_tRNA_optimization`
Runs the complete workflow:
- Trains all 4 variants for all pairs
- Evaluates all variants
- Runs comparison analysis
- Computes feature importance
- Tests sequence ablation
- Generates aggregate summary

```bash
snakemake --profile profiles/slurm all_tRNA_optimization
```

#### `all_tRNA_train`
Trains all variants without running analysis.

```bash
snakemake --profile profiles/slurm all_tRNA_train
```

#### `all_tRNA_compare`
Runs comparison analysis (assumes models are trained).

```bash
snakemake --profile profiles/slurm all_tRNA_compare
```

#### `tRNA_optimization_full_analysis`
Runs complete analysis for a single pair.

```bash
snakemake --profile profiles/slurm tRNA_optimization_full_analysis --config pair=charged_uncharged
```

## Configuration

### Required Configuration

The workflow requires the same configuration as standard leech pipelines:

```yaml
# pipeline/config/config.yaml
samples:
  charged_sample:
    pod5: "/path/to/charged.pod5"
    bam: "/path/to/charged.bam"
    label: "charged"
  uncharged_sample:
    pod5: "/path/to/uncharged.pod5"
    bam: "/path/to/uncharged.bam"
    label: "uncharged"

comparison_spec_file: "config/comparisons_charged_uncharged.tsv"

# Training parameters
epochs: 50
batch_size: 128
learning_rate: 0.001
early_stopping_patience: 5
```

### Comparison Spec File

Create a TSV file defining pairwise comparisons:

```tsv
# config/comparisons_charged_uncharged.tsv
charged	charged	uncharged	uncharged
```

For multi-label comparisons:

```tsv
# config/comparisons_amino_acids.tsv
Ala	Ala	Gly	Gly
basic	Arg,Lys	acidic	Asp,Glu
```

### Optional Configuration

Customize which variants to run:

```yaml
# Use all variants (default)
tRNA_variants:
  - base
  - dwell
  - tcn_signal_features
  - dwell_masked

# Or run a subset
tRNA_variants:
  - tcn_signal_features  # Only run the optimized variant
  - dwell                # And the standard baseline
```

## Usage Examples

### Example 1: Full Comparison (Charged vs Uncharged)

```bash
# 1. Prepare training data
snakemake --profile profiles/slurm all_prepare

# 2. Merge and split at read level
snakemake --profile profiles/slurm all_merge

# 3. Run complete tRNA optimization analysis
snakemake --profile profiles/slurm all_tRNA_optimization

# Check results
cat results/metrics/tRNA_optimization/aggregate/variant_summary.txt
```

### Example 2: Single Pair Analysis

```bash
# Train and analyze variants for a specific pair
snakemake --profile profiles/slurm \
    results/metrics/tRNA_optimization/pairwise/charged_uncharged/comparison_results.json

# View comparison results
cat results/metrics/tRNA_optimization/pairwise/charged_uncharged/comparison_summary.txt
```

### Example 3: Dry Run to Check Workflow

```bash
# See what will be executed
snakemake --profile profiles/slurm all_tRNA_optimization -n

# Print rule DAG
snakemake --dag all_tRNA_optimization | dot -Tpng > dag.png
```

### Example 4: Local Testing (Small Dataset)

```bash
# Run locally on a small test dataset
snakemake --cores 4 all_tRNA_train

# Compare results
snakemake --cores 2 all_tRNA_compare
```

### Example 5: CPU-Only Training

For clusters without GPU access:

```yaml
# config/config.yaml
use_cpu_training: true
```

```bash
snakemake --profile profiles/slurm all_tRNA_optimization
```

## Output Structure

```
results/
├── models/
│   └── tRNA_optimization/
│       └── pairwise/
│           └── charged_uncharged/
│               ├── base/
│               │   ├── model_best.pt
│               │   ├── model_last.pt
│               │   └── training_history.json
│               ├── dwell/
│               ├── tcn_signal_features/
│               └── dwell_masked/
└── metrics/
    └── tRNA_optimization/
        ├── pairwise/
        │   └── charged_uncharged/
        │       ├── comparison_results.json
        │       ├── comparison_summary.txt
        │       ├── training_curves.png
        │       ├── feature_importance/
        │       │   ├── importance_scores.npz
        │       │   └── importance_plot.png
        │       └── sequence_ablation/
        │           ├── ablation_results.json
        │           └── ablation_plot.png
        └── aggregate/
            ├── variant_comparison.tsv
            └── variant_summary.txt
```

## Expected Results

### Performance Rankings

Expected ranking (best to worst) for constant-sequence tRNA:

1. **TCNSignalFeatures**: Best - no sequence overfitting, optimized architecture
2. **ConvLSTMDwell with masking**: Good - masking prevents overfitting
3. **ConvLSTMDwell**: Fair - suffers from sequence overfitting
4. **ConvLSTMBase**: Baseline - no features, limited performance

### Performance Improvements

Typical improvements vs baseline (ConvLSTMBase):

- **TCNSignalFeatures**: +5-10% accuracy improvement
- **Masking**: +3-7% accuracy improvement
- **Standard Dwell**: +2-5% accuracy improvement (reduced by overfitting)

### Training Speed

Relative training times:

- **TCNSignalFeatures**: Fastest (30% fewer parameters than full models)
- **ConvLSTMSignalFeatures**: Fast (30% fewer parameters)
- **Dwell (masked)**: Same as standard
- **ConvLSTMDwell**: Baseline speed
- **ConvLSTMBase**: Slightly faster (no feature branch)

### Sequence Ablation Results

For models with sequence branches, ablation should show:

- **ConvLSTMDwell**: Large performance drop when sequence is randomized (indicates overfitting)
- **Dwell_masked**: Small performance drop (masking successfully prevents overfitting)

## Troubleshooting

### Issue: Config file not found

```
FileNotFoundError: configs/comparison/tcn_signal_features.yaml
```

**Solution**: Ensure you're running from the repository root and config files exist.

### Issue: TRNA_VARIANTS not defined

```
NameError: name 'TRNA_VARIANTS' is not defined
```

**Solution**: Make sure the tRNA_optimization.smk file is included in the main Snakefile.

### Issue: GPU out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use CPU training:

```yaml
batch_size: 64  # Reduce from default 128
# OR
use_cpu_training: true
```

### Issue: Models train but comparison fails

```
FileNotFoundError: model_best.pt
```

**Solution**: Ensure all 4 variants finished training before running comparison. Check logs in `results/models/tRNA_optimization/`.

## Integration with Existing Workflows

The tRNA optimization workflow is **independent** of the standard model comparison workflow:

- **Standard workflow** (`all_compare_models`): Compares architectures specified in `models_to_compare`
- **tRNA workflow** (`all_tRNA_optimization`): Compares 4 specific variants for tRNA optimization

You can run both workflows on the same data:

```bash
# Run standard architecture comparison
snakemake --profile profiles/slurm all_compare_models

# Run tRNA optimization comparison
snakemake --profile profiles/slurm all_tRNA_optimization
```

Results will be in different directories:
- Standard: `results/metrics/comparison/`
- tRNA: `results/metrics/tRNA_optimization/`

## Next Steps

After running the workflow:

1. **Review aggregate summary**: Check `results/metrics/tRNA_optimization/aggregate/variant_summary.txt`
2. **Compare training curves**: View `results/metrics/tRNA_optimization/pairwise/{pair}/training_curves.png`
3. **Analyze feature importance**: Examine which features drive predictions
4. **Test sequence ablation**: Quantify sequence branch contribution
5. **Choose best model**: Use TCNSignalFeatures for production tRNA classification

## References

- Implementation details: `docs/guides/09-TRNA_OPTIMIZATION_IMPLEMENTATION.md`
- Model selection guide: `docs/guides/10-MODEL_SELECTION_GUIDE.md`
- CLI documentation: `CLAUDE.md`
- Comparison configs: `configs/comparison/README.md`
