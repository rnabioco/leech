# Grid Search Usage Guide

This guide shows how to use the `leech model optimize` command to optimize chunk context parameters for model training.

## Overview

Grid search systematically tests different signal window sizes (left/right context) to find the optimal configuration for your model. This is essential because:
- Optimal context varies by amino acid pair
- Asymmetric contexts often perform best (e.g., 9500 left / 500 right)
- More context isn't always better (diminishing returns after ~10k samples)

## Quick Start

### 1. Prepare Training Data

First, prepare your training and validation chunks:

```bash title="Bash" linenums="1"
# Prepare training data (e.g., charged tRNAs)
uv run leech data prepare \
  --pod5 charged_reads.pod5 \
  --bam charged_alignments.bam \
  --output-dir data/train/ \
  --motif CCA \
  --motif-offset 2 \
  --label 1

# Prepare validation data
uv run leech data prepare \
  --pod5 val_charged_reads.pod5 \
  --bam val_charged_alignments.bam \
  --output-dir data/val/ \
  --motif CCA \
  --motif-offset 2 \
  --label 1
```

### 2. Run Grid Search

Run a coarse grid search to explore the parameter space:

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --model ConvLSTMDwell \
  --context-grid 200,500,1000,2000,5000 \
  --output-dir models/grid_coarse/ \
  --epochs 50 \
  --batch-size 128 \
  --device cuda
```

This will train 25 models (5 x 5 grid) and save results to `models/grid_coarse/`.

### 3. Analyze Results

The grid search produces a summary CSV:

```bash title="Bash" linenums="1"
# View results
cat models/grid_coarse/grid_summary.csv
```

Example output:
```
left_context,right_context,signal_len,best_val_acc,best_val_auc,best_epoch,train_time_sec,model_path
200,200,400,0.8923,0.9451,23,145.2,models/grid_coarse/left_200_right_200
200,500,700,0.9051,0.9623,28,182.5,models/grid_coarse/left_200_right_500
...
```

### 4. Fine-Tune Around Optimum (Optional)

If your best result is at the edge of the grid, run a fine-grained search.
Use range syntax (`start:stop:step`) for evenly-spaced grids:

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --model ConvLSTMDwell \
  --left-contexts 8000:10000:500 \
  --right-contexts 0:2000:500 \
  --output-dir models/grid_fine/ \
  --epochs 50 \
  --device cuda
```

This is equivalent to `--left-contexts 8000,8500,9000,9500,10000` and
`--right-contexts 0,500,1000,1500,2000`. The range stop is inclusive.

## Command Reference

### Basic Usage

```bash title="Bash" linenums="1"
leech model optimize \
  --train-data <path> \
  --val-data <path> \
  --context-grid <values> \
  --output-dir <path>
```

### Required Arguments

- `--train-data`: Path to training chunks (.npz file)
- `--context-grid`: Context values — comma-separated (e.g., "200,500,1000") or range as start:stop:step (e.g., "200:1000:200")
- `--output-dir`: Directory for grid search results

### Optional Arguments

- `--val-data`: Validation chunks (recommended for early stopping)
- `--model`: Model architecture (default: ConvLSTMDwell)
  - `ConvLSTMDwell`: Full model with dwell features
  - `ConvLSTMBase`: Baseline without dwell features
- `--left-contexts`: Override left context grid
- `--right-contexts`: Override right context grid
- `--kmer-context`: K-mer sequence context (default: 5)
- `--epochs`: Training epochs per grid point (default: 50)
- `--batch-size`: Batch size (default: 128)
- `--learning-rate`: Learning rate (default: 0.001)
- `--device`: Training device (cuda/cpu, default: cuda)
- `--parallel`: Number of grid points to train concurrently (default: 1)
- `--seed`: Random seed (default: 42)

## Parallel Execution

Grid search can train multiple grid points concurrently using `--parallel`:

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --context-grid 200,500,1000,2000,5000 \
  --output-dir models/grid_coarse/ \
  --epochs 50 \
  --device cpu \
  --parallel 8
```

Each worker process independently loads the training data once during initialization and caches it for all grid points it processes, avoiding redundant disk I/O.

**When to use parallel execution:**

- **CPU training**: `--parallel N` scales well since each worker gets its own CPU cores. Set N to roughly `total_cores / cores_per_model`.
- **GPU training**: Parallel execution is less useful since models compete for GPU memory. Use `--parallel 1` (default) with GPU, or `--parallel 2` if you have enough VRAM.

**CPU optimizations** (always active):

- Training data and validation data are pre-loaded once before the grid search begins
- Class weights are pre-computed and shared across all grid points
- DataLoader uses `num_workers=0` on CPU to avoid multiprocessing contention with the grid search pool

## Advanced Examples

### Asymmetric Grid Search

Test different left vs right contexts:

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --left-contexts 5000,7500,10000 \
  --right-contexts 200,500,1000 \
  --output-dir models/grid_asymmetric/ \
  --epochs 50
```

### Quick Exploration (Fewer Epochs)

For rapid prototyping, use fewer epochs:

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --context-grid 200,1000,5000 \
  --output-dir models/grid_quick/ \
  --epochs 10 \
  --device cuda
```

### CPU Training (No GPU Available)

```bash title="Bash" linenums="1"
uv run leech model optimize \
  --train-data data/train/chunks.npz \
  --val-data data/val/chunks.npz \
  --context-grid 200,500,1000 \
  --output-dir models/grid_cpu/ \
  --epochs 20 \
  --device cpu
```

## Output Structure

Grid search creates the following directory structure:

```
models/grid_coarse/
├── left_200_right_200/
│   ├── model_best.pt          # Best model checkpoint
│   ├── model_last.pt          # Final epoch checkpoint
│   ├── config.json            # Training configuration
│   ├── metrics.json           # Per-epoch metrics
│   └── summary.json           # Final summary stats
├── left_200_right_500/
│   └── ...
├── grid_config.json           # Grid search configuration
├── grid_summary.csv           # Aggregated results (ALL grid points)
└── best_params.json           # Best parameters for Snakemake integration
```

## Analyzing Results

### 1. Find Best Model

```bash title="Bash" linenums="1"
# Sort by validation accuracy
sort -t',' -k4 -r models/grid_coarse/grid_summary.csv | head -5
```

### 2. Visualize with R or Python

Example Python script:

```python title="Python" linenums="1"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("models/grid_coarse/grid_summary.csv")

# Create heatmap
pivot = df.pivot(index="right_context", columns="left_context", values="best_val_acc")

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Validation Accuracy by Chunk Context")
plt.xlabel("Left Context (samples, 3')")
plt.ylabel("Right Context (samples, 5')")
plt.savefig("grid_heatmap.png", dpi=300, bbox_inches="tight")
```

Example R script (from grid-search.md):

```r title="R" linenums="1"
library(ggplot2)
library(dplyr)

df <- read_csv("models/grid_coarse/grid_summary.csv")

ggplot(df, aes(x = left_context, y = right_context, fill = best_val_acc)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", best_val_acc)), size = 3) +
  scale_fill_viridis_c(option = "plasma", limits = c(0.5, 1.0)) +
  labs(title = "Validation Accuracy Across Chunk Contexts",
       x = "Left context (samples, 3')",
       y = "Right context (samples, 5')")
```

## Tips and Best Practices

### 1. Start with Coarse Grid

Begin with widely spaced values (200, 1000, 5000) to explore the parameter space efficiently.

### 2. Use Validation Data

Always provide validation data for early stopping and reliable model selection.

### 3. Watch for Overfitting

If validation accuracy plateaus while training continues to improve, consider:
- Reducing model capacity
- Adding dropout
- Using more training data

### 4. Computational Resources

- Each grid point trains a full model
- 5x5 grid = 25 models at ~3-5 min each = 1-2 hours total (GPU)
- Use `--parallel N` on CPU to train multiple grid points concurrently
- Use `--epochs 10` for quick exploration, then train final model longer

### 5. Biological Validation

After finding optimal parameters on synthetic data:
1. Train final model with optimal context
2. Validate on biological replicates
3. Check for clear separation between classes
4. Calibrate with mixture models (see grid-search.md)

## Troubleshooting

### Out of Memory Errors

If training fails with CUDA OOM:

```bash title="Bash" linenums="1"
# Reduce batch size
--batch-size 64

# Or reduce signal length (use smaller contexts)
--context-grid 200,500,1000
```

### Slow Training

- Use GPU (`--device cuda`)
- Use `--parallel N` to train multiple grid points concurrently on CPU
- Reduce number of grid points
- Reduce epochs for initial exploration

### Poor Performance

- Check data quality (sufficient reads, clear labels)
- Verify motif and offset are correct
- Try both ConvLSTMDwell and ConvLSTMBase
- Increase training data

## Integration with Snakemake

Example Snakemake rule:

```python title="Python" linenums="1"
rule grid_search:
    input:
        train = "data/train/chunks.npz",
        val = "data/val/chunks.npz"
    output:
        summary = "models/grid/grid_summary.csv"
    params:
        contexts = "200,500,1000,2000,5000",
        epochs = 50
    threads: 8
    resources:
        gpu = 1
    shell:
        """
        leech model optimize \
            --train-data {input.train} \
            --val-data {input.val} \
            --context-grid {params.contexts} \
            --output-dir models/grid/ \
            --epochs {params.epochs} \
            --device cuda \
            --parallel {threads}
        """
```

## See Also

- [Grid Search Strategy](grid-search.md) - Full methodology and statistical analysis
- [Architecture](../architecture.md) - Project overview and architecture
- [Getting Started](../getting-started/quick-start.md) - Installation and quick start
