# Grid Search and Model Evaluation Strategy for Leach

 

This document summarizes the model training exploration workflow developed through iterative analysis of Phe-Tyr and Thr-Ser pairwise amino acid models. It serves as a guide for implementing systematic model optimization and evaluation in the leach library.

 

## Overview

 

The workflow involves:

1. **Chunk context grid search** - Systematically test different signal window sizes

2. **Multi-level validation** - Synthetic → biological → test set evaluation

3. **Distribution-based quantification** - Avoid hard cutoffs, use mixture models

4. **Statistical rigor** - Bootstrap confidence intervals and permutation tests

 

---

 

## 1. Chunk Context Grid Search

 

### Purpose

Find the optimal signal window (left/right context) around the modification site that maximizes classification accuracy.

 

### Implementation Strategy

 

#### Grid Design

- **Tested contexts**: 200-10000 samples on each side of the motif

- **Common grids**:

  - Coarse: `[200, 500, 1000, 2000, 5000]` × `[200, 500, 1000, 2000, 5000]`

  - Fine (around optima): `[8000, 8500, 9000, 9500, 10000]` × `[0, 500, 1000, 1500, 2000]`

 

#### Training Protocol

```bash

# Example: Train models across grid

for left in 200 500 1000 2000 5000; do

  for right in 200 500 1000 2000 5000; do

    leach train \

      --train-data train.json \

      --val-data val.json \

      --model ConvLSTMDwell \

      --chunk-context $left $right \

      --output-dir models/train_${left}_${right}

  done

done

```

 

#### Evaluation Metrics

1. **Validation accuracy** - Primary metric during training

2. **Final epoch performance** - Use last or best checkpoint

3. **Heatmap visualization** - Plot accuracy across (left, right) grid

 

#### Key Findings

- **Asymmetric contexts often optimal**: e.g., 9500 left / 500 right for Phe-Tyr

- **Biological data may differ from synthetic**: Always validate on both

- **Diminishing returns**: Beyond ~10000 samples, gains are minimal

 

### Visualization Code Pattern

```r

# R code for heatmap

library(ggplot2)

 

df <- read_csv("grid_search_results.csv")

 

ggplot(df, aes(x = left, y = right, fill = val_acc)) +

  geom_tile(color = "white") +

  geom_text(aes(label = sprintf("%.2f", val_acc)), size = 3) +

  scale_fill_viridis_c(option = "plasma", limits = c(0.5, 1.0)) +

  labs(title = "Validation Accuracy Across Chunk Contexts",

       x = "Left context (samples, 3')",

       y = "Right context (samples, 5')")

```

 

---

 

## 2. Multi-Level Validation Strategy

 

### Validation Hierarchy

 

#### Level 1: Synthetic Data Training

- **Purpose**: Establish proof-of-concept with clean signal

- **Data**: Pure Phe/Tyr or Thr/Ser tRNAs from controlled experiments

- **Split**: 70% train / 15% validation / 15% test

- **Metric**: Binary classification accuracy

 

#### Level 2: Biological Validation

- **Purpose**: Confirm model generalizes to biological complexity

- **Data**: Wild-type yeast grown in defined media (e.g., SC, YPD)

- **Expected**: Clear separation between true Phe vs Tyr (or Thr vs Ser)

- **Visualization**: Density plots with Okabe-Ito colorblind-safe palette

 

```r

# Expected pattern for biological validation

aa_cols <- c(Phe = "#009E73", Tyr = "#F0E442")

 

ggplot(df, aes(x = ML_val, fill = AA)) +

  geom_density(alpha = 0.5) +

  scale_fill_manual(values = aa_cols) +

  facet_wrap(~replicate, ncol = 1) +

  labs(x = "Modification likelihood", y = "Density")

```

 

#### Level 3: Test Set Evaluation

- **Purpose**: Unbiased performance estimate

- **Critical**: Never use test set for hyperparameter tuning

- **Metrics**: Accuracy, ROC-AUC, calibration curves

 

---

 

## 3. Quantification Methods for Misaminoacylation

 

### Problem Statement

Given a population of tRNA^X reads, estimate the fraction mis-charged with amino acid Y.

 

### Method Comparison

 

#### A. Mean ML (Simple Average)

```python

def estimate_mean_ml(ml_values):

    """Mean of modification likelihoods"""

    return np.mean(ml_values)

```

 

**Pros**: Simple, intuitive

**Cons**: Biased, especially at low fractions (<5%)

**LoD**: ~2%

 

#### B. MLE (Maximum Likelihood Estimator) — RECOMMENDED

```python

def estimate_mle(ml_values, f0_density, f1_density):

    """

    Mixture model MLE on logit scale.

 

    Args:

        ml_values: Observed ML scores (0-1 scale)

        f0_density: KDE of canonical class (e.g., Phe)

        f1_density: KDE of modified class (e.g., Tyr)

 

    Returns:

        Estimated fraction of modified class

    """

    # Transform to logit space

    logits = scipy.special.logit(np.clip(ml_values, 1e-6, 1-1e-6))

 

    # Mixture likelihood

    def neg_log_lik(pi):

        mix = (1 - pi) * f0_density(logits) + pi * f1_density(logits)

        return -np.sum(np.log(mix))

 

    result = scipy.optimize.minimize_scalar(

        neg_log_lik, bounds=(0, 1), method='bounded'

    )

    return result.x

```

 

**Pros**: Unbiased, better at low fractions

**Cons**: Requires pre-fit densities from pure classes

**LoD**: ~1%

 

### Calibration Procedure

 

#### Step 1: Build Reference Densities (Training Phase)

```python

# Use TEST set pure Phe and Tyr to learn class shapes

phe_test = test_df[test_df['aa'] == 'Phe']['ML_val']

tyr_test = test_df[test_df['aa'] == 'Tyr']['ML_val']

 

# Fit KDEs on logit scale

from scipy.stats import gaussian_kde

 

phe_logits = logit(np.clip(phe_test, 1e-6, 1-1e-6))

tyr_logits = logit(np.clip(tyr_test, 1e-6, 1-1e-6))

 

kde_phe = gaussian_kde(phe_logits, bw_method='scott')

kde_tyr = gaussian_kde(tyr_logits, bw_method='scott')

```

 

#### Step 2: Create In Silico Mixtures

```python

def make_mixture(pi_true, n_reads=2000):

    """Draw synthetic mixture at known pi_true"""

    n_tyr = int(pi_true * n_reads)

    n_phe = n_reads - n_tyr

 

    mixture = np.concatenate([

        np.random.choice(phe_test, n_phe, replace=True),

        np.random.choice(tyr_test, n_tyr, replace=True)

    ])

    return mixture

 

# Test grid: 0-10% Tyr in 0.5% steps

grid = np.arange(0, 0.11, 0.005)

```

 

#### Step 3: Bootstrap LoD/LoQ

```python

B = 500  # bootstrap replicates

results = []

 

for pi_true in grid:

    for b in range(B):

        mix = make_mixture(pi_true)

        mle_est = estimate_mle(mix, kde_phe, kde_tyr)

        mean_ml_est = estimate_mean_ml(mix)

 

        results.append({

            'true_pct': pi_true * 100,

            'mle': mle_est * 100,

            'mean_ml': mean_ml_est * 100

        })

 

results_df = pd.DataFrame(results)

```

 

#### Step 4: Define LoD and LoQ

 

**Limit of Detection (LoD)**:

- Minimum % where 95% CI does NOT overlap with null (0%)

- Typical: 1-2% for good models

 

**Limit of Quantification (LoQ)**:

- RMSE ≤ 1.0 absolute percentage points, OR

- Relative error ≤ 20%, OR

- 95% CI width ≤ 2 percentage points

- Typical: 2-5% for good models

 

```r

# LoD calculation (R)

null_p95 <- results_df %>%

  filter(true_pct == 0) %>%

  group_by(estimator) %>%

  summarise(cutoff = quantile(est_pct, 0.95))

 

lod <- results_df %>%

  group_by(estimator, true_pct) %>%

  summarise(lo95 = quantile(est_pct, 0.025)) %>%

  filter(lo95 > null_p95) %>%

  slice_min(true_pct)

```

 

---

 

## 4. Distribution-Based Evaluation Metrics

 

### Why Avoid Hard Cutoffs?

- **Data-dependent**: Optimal threshold varies by model and dataset

- **Information loss**: Binary classification discards continuous information

- **Overfitting risk**: Tuning cutoff on validation set inflates performance

 

### Recommended Metrics

 

#### A. Jensen-Shannon Divergence

Measures distributional difference between two groups (bits).

 

```python

from scipy.spatial.distance import jensenshannon

 

def js_divergence(group_a, group_b, bins=50):

    """

    Compute JS divergence between two distributions.

 

    Args:

        group_a, group_b: ML value arrays

        bins: Number of histogram bins

 

    Returns:

        JS divergence in bits

    """

    # Shared bins

    all_vals = np.concatenate([group_a, group_b])

    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins+1)

 

    # Histograms with Laplace smoothing

    ha, _ = np.histogram(group_a, bins=bin_edges)

    hb, _ = np.histogram(group_b, bins=bin_edges)

 

    pa = (ha + 1e-3) / (ha.sum() + 1e-3 * bins)

    pb = (hb + 1e-3) / (hb.sum() + 1e-3 * bins)

 

    return jensenshannon(pa, pb, base=2)

```

 

**Usage**: Compare experimental group to wild-type baseline

- JS ≈ 0: Distributions identical

- JS > (WT_mean + 2×SE): Significantly different

 

#### B. ΔECDF (Difference in Empirical CDFs)

Shows exactly where distributions differ.

 

```python

def delta_ecdf(target, baseline, grid=None):

    """

    Compute ΔECDF = ECDF_target - ECDF_baseline.

 

    Returns:

        x_grid, delta_cdf

    """

    if grid is None:

        grid = np.linspace(

            min(target.min(), baseline.min()),

            max(target.max(), baseline.max()),

            500

        )

 

    ecdf_target = np.searchsorted(np.sort(target), grid, side='right') / len(target)

    ecdf_base = np.searchsorted(np.sort(baseline), grid, side='right') / len(baseline)

 

    return grid, ecdf_target - ecdf_base

```

 

**Interpretation**:

- Positive ΔECDF: Target has more high-scoring reads

- Negative ΔECDF: Target has fewer high-scoring reads

- Plot with bootstrap CI across replicates

 

```r

# Visualization with ggplot2

ggplot(delta_df, aes(x = ML_val, y = delta_ecdf, color = group)) +

  geom_hline(yintercept = 0, linetype = "dashed") +

  geom_line(aes(group = interaction(group, replicate)), alpha = 0.3) +

  geom_ribbon(data = group_means,

              aes(ymin = lo95, ymax = hi95, fill = group),

              alpha = 0.2) +

  labs(x = "ML value", y = "ΔECDF vs WT baseline")

```

 

#### C. Binned Percentage Plots

Show *where* in the ML distribution the signal appears.

 

```r

# Bin ML values and compute % per replicate

df_binned <- df %>%

  mutate(bin = cut(ML_val, breaks = seq(0, 255, by = 10))) %>%

  group_by(genotype, media, replicate, AA, bin) %>%

  summarise(n = n()) %>%

  group_by(genotype, media, replicate, AA) %>%

  mutate(pct = 100 * n / sum(n))

 

# Average across replicates

df_summary <- df_binned %>%

  group_by(genotype, media, AA, bin) %>%

  summarise(mean_pct = mean(pct),

            se = sd(pct) / sqrt(n()))

 

# Plot with ribbon for uncertainty

ggplot(df_summary, aes(x = bin, y = mean_pct, color = AA)) +

  geom_line() +

  geom_ribbon(aes(ymin = mean_pct - se, ymax = mean_pct + se, fill = AA),

              alpha = 0.2) +

  facet_grid(genotype ~ media) +

  labs(y = "% of reads")

```

 

---

 

## 5. Integration with Leach Workflow

 

### Proposed Module Structure

 

```

leach/

├── src/leach/

│   ├── gridsearch.py      # NEW: Chunk context grid search

│   ├── calibration.py     # NEW: MLE estimator + LoD/LoQ

│   ├── evaluation.py      # EXPAND: Add JS, ΔECDF metrics

│   └── visualization.py   # NEW: Standard plots

└── workflow/

    └── Snakefile_gridsearch  # NEW: Grid search pipeline

```

 

### Snakemake Rule Example

 

```python

rule grid_search_train:

    input:

        train = "data/splits/train.json",

        val = "data/splits/val.json"

    output:

        model = "models/grid/{left}_{right}/model_best.pt",

        metrics = "models/grid/{left}_{right}/metrics.json"

    params:

        left = lambda wc: wc.left,

        right = lambda wc: wc.right

    shell:

        """

        leach train \

            --train-data {input.train} \

            --val-data {input.val} \

            --chunk-context {params.left} {params.right} \

            --output-dir $(dirname {output.model})

        """

 

rule summarize_grid:

    input:

        expand("models/grid/{left}_{right}/metrics.json",

               left=[200, 500, 1000, 2000, 5000],

               right=[200, 500, 1000, 2000, 5000])

    output:

        summary = "results/grid_search_summary.csv",

        heatmap = "results/grid_search_heatmap.pdf"

    script:

        "scripts/summarize_grid.py"

```

 

---

 

## 6. Statistical Testing Guidelines

 

### Biological Replicates

- **Minimum**: n=3 per condition

- **Report**: Mean ± SE across replicates

- **Plot**: Show individual replicates + group mean

 

### Permutation Tests

For comparing conditions (e.g., MT+serine vs MT+minimal):

 

```python

def permutation_test(group_a, group_b, metric_fn, n_perm=10000):

    """

    Permutation test for difference in metric.

 

    Args:

        group_a, group_b: Arrays of replicate-level metrics

        metric_fn: Function to compute group difference

        n_perm: Number of permutations

 

    Returns:

        observed_diff, p_value

    """

    observed = metric_fn(group_a) - metric_fn(group_b)

 

    combined = np.concatenate([group_a, group_b])

    n_a = len(group_a)

 

    null_diffs = []

    for _ in range(n_perm):

        perm = np.random.permutation(combined)

        diff = metric_fn(perm[:n_a]) - metric_fn(perm[n_a:])

        null_diffs.append(diff)

 

    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed))

    return observed, p_value

```

 

### Bootstrap Confidence Intervals

- **Replicates**: 500-2000 bootstrap samples

- **Method**: Percentile method (2.5th, 97.5th percentiles)

- **Stratified**: When combining replicates, resample *replicates* not reads

 

---

 

## 7. Recommended Workflow for New Pairwise Models

 

### Phase 1: Initial Exploration (Synthetic Data)

1. Train baseline models at 200/200 and 500/500 contexts

2. Verify clear separation between classes (AUC > 0.95)

3. If poor: Check data quality, alignment, motif specification

 

### Phase 2: Grid Search (Synthetic Data)

1. Coarse grid: 200-5000 in 4-5 steps

2. Identify high-performing region

3. Fine grid: ±20% around optimum in 500-1000 sample steps

4. Select model with highest validation accuracy

 

### Phase 3: Biological Validation

1. Run inference on 3+ biological replicates of pure classes

2. Verify distribution separation

3. Fit KDEs for MLE calibration

4. Establish LoD/LoQ via bootstrap

 

### Phase 4: Experimental Application

1. Use MLE estimator for mixture quantification

2. Report estimates with 95% CI

3. Use JS divergence for group comparisons

4. Permutation test for significance (p < 0.05)

 

---

 

## 8. Quality Control Checklist

 

Before deploying a model to production:

 

- [ ] Validation accuracy > 0.90

- [ ] AUC on test set > 0.95

- [ ] LoD ≤ 2% (MLE estimator)

- [ ] LoQ ≤ 5% (RMSE criterion)

- [ ] Biological validation shows clear separation

- [ ] Calibration curve is linear (slope ≈ 1)

- [ ] Tested on ≥3 biological replicates

- [ ] JS divergence detects known misaminoacylation (if available)

 

---

 

## 9. Key Takeaways

 

1. **Chunk context matters**: Grid search is essential, optimal windows are often asymmetric

2. **Avoid hard cutoffs**: Use distribution-based metrics (JS divergence, ΔECDF)

3. **MLE > mean ML**: For quantifying mixtures, especially at low fractions

4. **Bootstrap everything**: LoD, LoQ, confidence intervals

5. **Biological validation**: Synthetic training must generalize to biological data

6. **Replicate, replicate, replicate**: n≥3 for statistical rigor

 

---

 

## 10. References to Original Analyses

 

- Phe-Tyr grid search: `20250912_dataprep.qmd`

- Ibba strain screening: `20250917_ibba_ml_screening.qmd`

- Thr-Ser training: `thr_ser_training.qmd`

- Phe-Tyr training: `phe_tyr_training.qmd`

- Misaminoacylation quantification: `202509_misacylation.qmd`

 

All original notebooks contain detailed R code for visualization and statistical testing that can be adapted for leach's Python implementation.
