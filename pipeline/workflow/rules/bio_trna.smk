"""
Biological tRNA Discriminative Models Workflow
================================================

This module implements a workflow for training discriminative models from
biological tRNA samples. The workflow:

1. Aligns biological tRNA reads to a reference database to identify isodecoder/amino acid
2. Classifies reads by charging status using a pre-trained charged/uncharged model
3. Splits reads by amino acid and charging status
4. Trains two types of discriminative models:
   a. Pairwise amino acid discrimination (Ala vs Arg, Ala vs Asn, etc.)
   b. Per-amino acid charging discrimination (Ala-charged vs Ala-uncharged, etc.)

Usage:
    # Run alignment and classification
    snakemake --profile profiles/slurm all_bio_align

    # Prepare chunks for biological samples
    snakemake --profile profiles/slurm all_bio_prepare

    # Train pairwise AA models
    snakemake --profile profiles/slurm all_bio_train_pairwise_aa

    # Train per-AA charging models
    snakemake --profile profiles/slurm all_bio_train_charging
"""

from pathlib import Path
import re
import pysam


# ============================================================================
# Helper Functions
# ============================================================================

def parse_aa_from_reference(ref_name):
    """Parse amino acid from tRNA reference name.

    Expected format: tRNA-Ala-AGC-1-1, tRNA-Arg-ACG-1-1, etc.
    Returns the 3-letter amino acid code (e.g., 'Ala', 'Arg').

    Args:
        ref_name: Reference sequence name

    Returns:
        str: 3-letter amino acid code or None if parsing fails
    """
    match = re.match(r'tRNA-([A-Z][a-z]{2})', ref_name)
    if match:
        return match.group(1)
    return None


def get_bio_amino_acids(wildcards):
    """Get list of amino acids present in biological sample.

    Reads the BAM file to extract amino acids from aligned references.
    This is computed dynamically based on what's in the aligned BAM.

    Args:
        wildcards: Snakemake wildcards (must contain 'sample')

    Returns:
        list: Unique 3-letter amino acid codes found in the sample
    """
    bam_path = (
        get_project_path(config.get("bio_bam_dir", "results/bam/bio"))
        + f"/{wildcards.sample}/{wildcards.sample}.aligned.bam"
    )

    if not Path(bam_path).exists():
        return []

    amino_acids = set()
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for ref_name in bam.references:
                aa = parse_aa_from_reference(ref_name)
                if aa:
                    amino_acids.add(aa)
    except:
        pass

    return sorted(amino_acids)


def get_bio_aa_charging_groups(sample):
    """Get all AA+charging groups for a biological sample.

    Returns groups like: Ala_charged, Ala_uncharged, Arg_charged, etc.

    Args:
        sample: Sample name

    Returns:
        list: Group names in format {AA}_{charged|uncharged}
    """
    # Get amino acids from config or dynamically from BAM
    bio_samples = config.get("bio_samples", {})
    if sample in bio_samples and "amino_acids" in bio_samples[sample]:
        amino_acids = bio_samples[sample]["amino_acids"]
    else:
        # Try to get from aligned BAM file
        class FakeWildcards:
            pass
        wc = FakeWildcards()
        wc.sample = sample
        amino_acids = get_bio_amino_acids(wc)

    groups = []
    for aa in amino_acids:
        groups.extend([f"{aa}_charged", f"{aa}_uncharged"])

    return groups


# ============================================================================
# Configuration Variables
# ============================================================================

# Biological sample directories
BIO_BAM_DIR = get_project_path(config.get("bio_bam_dir", "results/bam/bio"))
BIO_CHUNKS_DIR = get_project_path(config.get("bio_chunks_dir", "results/chunks/bio"))
BIO_MODELS_DIR = get_project_path(config.get("bio_models_dir", "results/models/bio"))
BIO_INFER_DIR = get_project_path(config.get("bio_inference_dir", "results/inference/bio"))
BIO_METRICS_DIR = get_project_path(config.get("bio_metrics_dir", "results/metrics/bio"))

# Biological samples (handle None when all entries are commented out)
BIO_SAMPLES = list((config.get("bio_samples") or {}).keys())

# tRNA reference database
TRNA_REFERENCE = config.get("trna_reference", "pipeline/resources/references/trna-reference.fa")

# Pre-trained charged/uncharged model for classification
CHARGING_MODEL = config.get("charging_model", "results/models/pairwise/charged_uncharged/model_best.pt")

# Biological comparison specs
BIO_AA_PAIRS = []
BIO_AA_PAIR_TO_LABELS = {}
BIO_CHARGING_PAIRS = []
BIO_AMINO_ACIDS = []

# Parse biological pairwise AA comparison spec
bio_aa_spec_file = config.get("bio_aa_comparison_spec_file")
if bio_aa_spec_file and Path(bio_aa_spec_file).exists():
    with open(bio_aa_spec_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            meta1, labels1_str, meta2, labels2_str = parts
            labels1 = [l.strip() for l in labels1_str.split(",")]
            labels2 = [l.strip() for l in labels2_str.split(",")]
            pair_name = f"{meta1}_{meta2}"
            BIO_AA_PAIRS.append(pair_name)
            BIO_AA_PAIR_TO_LABELS[pair_name] = (labels1, labels2)

# Parse biological per-AA charging comparison spec
# For per-AA charging, we extract unique amino acids from the spec
BIO_AMINO_ACIDS = []
bio_charging_spec_file = config.get("bio_charging_comparison_spec_file")
if bio_charging_spec_file and Path(bio_charging_spec_file).exists():
    aa_set = set()
    with open(bio_charging_spec_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            meta1, labels1_str, meta2, labels2_str = parts
            labels1 = [l.strip() for l in labels1_str.split(",")]
            labels2 = [l.strip() for l in labels2_str.split(",")]
            # Extract amino acids from labels (both labels should be the same AA)
            aa_set.update(labels1)
            aa_set.update(labels2)
    BIO_AMINO_ACIDS = sorted(aa_set)
    BIO_CHARGING_PAIRS = BIO_AMINO_ACIDS  # Use amino acids directly as pairs


# ============================================================================
# Rules
# ============================================================================

rule align_bio_trna_to_reference:
    """Align biological tRNA reads to tRNA reference database.

    This identifies the isodecoder/amino acid for each read.
    Uses minimap2 with nanopore preset for alignment.
    """
    input:
        bam=lambda wildcards: config["bio_samples"][wildcards.sample]["bam"],
        reference=TRNA_REFERENCE,
    output:
        bam=BIO_BAM_DIR + "/{sample}/{sample}.aligned.bam",
        bai=BIO_BAM_DIR + "/{sample}/{sample}.aligned.bam.bai",
    params:
        minimap2_bin=config.get("minimap2_bin", "minimap2"),
        samtools_bin=config.get("samtools_bin", "samtools"),
        minimap2_opts="-ax map-ont --secondary=no -L",  # nanopore preset, no secondary, long cigar
    threads: config.get("workers", 8)
    log:
        BIO_BAM_DIR + "/{sample}/align.log",
    shell:
        """
        # Extract FASTQ from input BAM, align to tRNA reference, sort and index
        {params.samtools_bin} fastq -T '*' {input.bam} | \
            {params.minimap2_bin} {params.minimap2_opts} -t {threads} \
            {input.reference} - | \
            {params.samtools_bin} sort -@ {threads} -o {output.bam} - \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam} 2>&1 | tee -a {log}
        """


rule infer_charging_status:
    """Classify reads by charging status using pre-trained model.

    Uses leech inference with a trained charged/uncharged model to predict
    the charging status of each read. Outputs BAM with ML tags containing
    charging probabilities.
    """
    input:
        pod5=lambda wildcards: config["bio_samples"][wildcards.sample]["pod5"],
        bam=BIO_BAM_DIR + "/{sample}/{sample}.aligned.bam",
        model=CHARGING_MODEL,
    output:
        bam=BIO_BAM_DIR + "/{sample}/{sample}.charging_predictions.bam",
        bai=BIO_BAM_DIR + "/{sample}/{sample}.charging_predictions.bam.bai",
    params:
        batch_size=config.get("infer_batch_size", 256),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        samtools_bin=config.get("samtools_bin", "samtools"),
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=lambda wildcards, attempt: 240,
        cpus_per_task=lambda wildcards, attempt: 8,
        mem_mb=32000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        BIO_BAM_DIR + "/{sample}/infer_charging.log",
    shell:
        """
        uv run leech infer \
            --model {input.model} \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output {output.bam} \
            --batch-size {params.batch_size} \
            --device {params.device} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam} 2>&1 | tee -a {log}
        """


rule split_bam_by_aa_and_charging:
    """Split BAM file by amino acid and charging status.

    Parses the aligned reference names to extract amino acid identity,
    uses ML tags from charging inference to classify charged vs uncharged,
    and splits the BAM into separate files for each AA+charging combination.

    Outputs: {sample}/Ala_charged.bam, {sample}/Ala_uncharged.bam, etc.
    """
    input:
        bam=BIO_BAM_DIR + "/{sample}/{sample}.charging_predictions.bam",
        bai=BIO_BAM_DIR + "/{sample}/{sample}.charging_predictions.bam.bai",
    output:
        # Dynamic output - will create files for each AA+charging group found
        stamp=BIO_BAM_DIR + "/{sample}/split.done",
    params:
        output_dir=BIO_BAM_DIR + "/{sample}",
        threshold=config.get("charging_threshold", 0.5),  # Probability threshold for charged classification
    log:
        BIO_BAM_DIR + "/{sample}/split.log",
    run:
        import pysam
        from pathlib import Path
        from collections import defaultdict

        # Open input BAM
        input_bam = pysam.AlignmentFile(input.bam, "rb")

        # Create output BAM files for each AA+charging group
        output_bams = {}

        reads_processed = 0
        reads_split = defaultdict(int)

        for read in input_bam:
            reads_processed += 1

            # Skip unmapped reads
            if read.is_unmapped:
                continue

            # Extract amino acid from reference name
            ref_name = input_bam.get_reference_name(read.reference_id)
            aa = parse_aa_from_reference(ref_name)

            if not aa:
                continue

            # Extract charging probability from ML tag
            # ML tag format: probability of class 1 (charged)
            try:
                ml_tag = read.get_tag("ML")
                charged_prob = float(ml_tag)
            except KeyError:
                # No ML tag - skip this read
                continue

            # Classify as charged or uncharged based on threshold
            charging_status = "charged" if charged_prob >= params.threshold else "uncharged"

            # Create group name
            group = f"{aa}_{charging_status}"

            # Open output BAM for this group if not already open
            if group not in output_bams:
                output_path = Path(params.output_dir) / f"{group}.bam"
                output_bams[group] = pysam.AlignmentFile(
                    str(output_path), "wb", header=input_bam.header
                )

            # Write read to appropriate output BAM
            output_bams[group].write(read)
            reads_split[group] += 1

        # Close all files
        input_bam.close()
        for bam in output_bams.values():
            bam.close()

        # Index all output BAMs
        import subprocess
        samtools_bin = config.get("samtools_bin", "samtools")
        for group in output_bams.keys():
            bam_path = Path(params.output_dir) / f"{group}.bam"
            subprocess.run([samtools_bin, "index", str(bam_path)])

        # Write completion stamp with summary
        with open(output.stamp, "w") as f:
            f.write(f"Processed {reads_processed} reads\n")
            f.write(f"Split into {len(output_bams)} groups:\n")
            for group in sorted(reads_split.keys()):
                f.write(f"  {group}: {reads_split[group]} reads\n")

        # Also write to log
        with open(log[0], "w") as f:
            f.write(f"Processed {reads_processed} reads\n")
            f.write(f"Split into {len(output_bams)} groups:\n")
            for group in sorted(reads_split.keys()):
                f.write(f"  {group}: {reads_split[group]} reads\n")


rule prepare_bio_chunks:
    """Extract training chunks for biological tRNA by AA+charging group.

    For each AA+charging combination (e.g., Ala_charged, Ala_uncharged),
    extracts training chunks from POD5 and split BAM files.
    """
    input:
        pod5=lambda wildcards: config["bio_samples"][wildcards.sample]["pod5"],
        bam=BIO_BAM_DIR + "/{sample}/{group}.bam",
        bai=BIO_BAM_DIR + "/{sample}/{group}.bam.bai",
        stamp=BIO_BAM_DIR + "/{sample}/split.done",
    output:
        all=BIO_CHUNKS_DIR + "/{sample}/{group}/all.npz",
    threads: config.get("workers", 8)
    params:
        output_dir=BIO_CHUNKS_DIR + "/{sample}/{group}",
        motif=config.get("motif", "CCATGGC"),
        motif_offset=config.get("motif_offset", 3),
        motif_reference=config.get("motif_reference", "fasta"),
        reference_fasta=TRNA_REFERENCE,
        skip_motif_indels=config.get("skip_motif_indels", True),
        workers=config.get("workers", 8),
        chunk_size=config.get("chunk_size", 100),
        # Extract label from group name (e.g., Ala_charged -> Ala for label)
        label=lambda wildcards: wildcards.group,  # Use full group name as label
        ref_fasta_arg=f"--reference-fasta {TRNA_REFERENCE}",
        skip_indels_arg=lambda wildcards: (
            "--skip-motif-indels" if config.get("skip_motif_indels", True) else ""
        ),
    log:
        BIO_CHUNKS_DIR + "/{sample}/{group}/prepare.log",
    shell:
        """
        uv run leech prepare \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --motif-reference {params.motif_reference} \
            {params.ref_fasta_arg} \
            {params.skip_indels_arg} \
            --label {params.label} \
            --workers {params.workers} \
            --chunk-size {params.chunk_size} \
            --no-split \
            2>&1 | tee {log}
        """


rule merge_bio_chunks_pairwise_aa:
    """Merge chunks for pairwise amino acid discrimination.

    Merges all charged+uncharged reads for two amino acids to train a
    discriminative model (e.g., all Ala reads vs all Arg reads).

    This allows the model to discriminate between amino acids regardless
    of charging status.
    """
    input:
        chunks=lambda wildcards: [
            BIO_CHUNKS_DIR + f"/{sample}/{aa}_{charging}/all.npz"
            for sample in BIO_SAMPLES
            for aa in BIO_AA_PAIR_TO_LABELS[wildcards.pair][0] + BIO_AA_PAIR_TO_LABELS[wildcards.pair][1]
            for charging in ["charged", "uncharged"]
        ],
    output:
        train=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/train.npz",
        val=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/val.npz",
        test=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/test.npz",
    params:
        output_dir=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        # Build input arguments: map AA_charged and AA_uncharged -> meta label
        input_args=lambda wildcards, input: build_bio_pairwise_aa_input_args(
            wildcards.pair, input.chunks
        ),
    log:
        BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/merge_and_split.log",
    shell:
        """
        uv run leech merge-and-split \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            2>&1 | tee {log}
        """


rule merge_bio_chunks_per_aa_charging:
    """Merge chunks for per-amino acid charging discrimination.

    Merges charged vs uncharged reads for a specific amino acid to train
    a charging status classifier (e.g., Ala-charged vs Ala-uncharged).

    This allows the model to discriminate charging status within a
    specific amino acid type.
    """
    input:
        chunks=lambda wildcards: [
            BIO_CHUNKS_DIR + f"/{sample}/{wildcards.aa}_{charging}/all.npz"
            for sample in BIO_SAMPLES
            for charging in ["charged", "uncharged"]
        ],
    output:
        train=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/train.npz",
        val=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/val.npz",
        test=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/test.npz",
    params:
        output_dir=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        # Build input arguments: map AA_charged -> charged, AA_uncharged -> uncharged
        input_args=lambda wildcards, input: build_bio_charging_input_args(
            wildcards.aa, input.chunks
        ),
    log:
        BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/merge_and_split.log",
    shell:
        """
        uv run leech merge-and-split \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            2>&1 | tee {log}
        """


rule train_bio_pairwise_aa:
    """Train pairwise amino acid discrimination model.

    Trains a model to discriminate between two amino acids, using reads
    from both charged and uncharged samples.
    """
    input:
        train=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/train.npz",
        val=BIO_CHUNKS_DIR + "/merged/pairwise_aa/{pair}/val.npz",
    output:
        model=BIO_MODELS_DIR + "/pairwise_aa/{pair}/model_best.pt",
        checkpoint=BIO_MODELS_DIR + "/pairwise_aa/{pair}/model_last.pt",
        history=BIO_MODELS_DIR + "/pairwise_aa/{pair}/metrics.json",
    params:
        output_dir=BIO_MODELS_DIR + "/pairwise_aa/{pair}",
        model_type=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=lambda wildcards, attempt: (
            480 if config.get("use_cpu_training", False) else 60
        ),
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 10
        ),
        mem_mb=18000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        BIO_MODELS_DIR + "/pairwise_aa/{pair}/train.log",
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule train_bio_per_aa_charging:
    """Train per-amino acid charging discrimination model.

    Trains a model to discriminate charged vs uncharged status for a
    specific amino acid.
    """
    input:
        train=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/train.npz",
        val=BIO_CHUNKS_DIR + "/merged/per_aa_charging/{aa}/val.npz",
    output:
        model=BIO_MODELS_DIR + "/per_aa_charging/{aa}/model_best.pt",
        checkpoint=BIO_MODELS_DIR + "/per_aa_charging/{aa}/model_last.pt",
        history=BIO_MODELS_DIR + "/per_aa_charging/{aa}/metrics.json",
    params:
        output_dir=BIO_MODELS_DIR + "/per_aa_charging/{aa}",
        model_type=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=lambda wildcards, attempt: (
            480 if config.get("use_cpu_training", False) else 60
        ),
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 10
        ),
        mem_mb=18000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        BIO_MODELS_DIR + "/per_aa_charging/{aa}/train.log",
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            2>&1 | tee {log}
        """


# ============================================================================
# Helper Functions for Input Arguments
# ============================================================================

def build_bio_pairwise_aa_input_args(pair, chunk_files):
    """Build -i label=file arguments for pairwise AA merge.

    Maps AA_charged and AA_uncharged groups to their meta labels (AA1 or AA2).

    Args:
        pair: Pair name (e.g., "Ala_Arg")
        chunk_files: List of chunk file paths

    Returns:
        String of -i arguments
    """
    labels1, labels2 = BIO_AA_PAIR_TO_LABELS[pair]
    meta1, meta2 = pair.split("_", 1)

    args = []
    for chunk_file in chunk_files:
        # Extract group from path: .../sample/Ala_charged/all.npz -> Ala_charged
        parts = Path(chunk_file).parts
        group = parts[-2]  # Group is second-to-last part

        # Parse AA from group (Ala_charged -> Ala)
        aa = group.rsplit("_", 1)[0]

        # Determine meta label
        if aa in labels1:
            meta_label = meta1
        elif aa in labels2:
            meta_label = meta2
        else:
            continue

        args.append(f"-i {meta_label}={chunk_file}")

    return " ".join(args)


def build_bio_charging_input_args(aa, chunk_files):
    """Build -i label=file arguments for per-AA charging merge.

    Maps AA_charged -> "charged" and AA_uncharged -> "uncharged".

    Args:
        aa: Amino acid (e.g., "Ala")
        chunk_files: List of chunk file paths

    Returns:
        String of -i arguments
    """
    args = []
    for chunk_file in chunk_files:
        # Extract group from path: .../sample/Ala_charged/all.npz -> Ala_charged
        parts = Path(chunk_file).parts
        group = parts[-2]

        # Parse charging status from group (Ala_charged -> charged)
        if "_" in group:
            charging = group.rsplit("_", 1)[1]  # Get last part after underscore
            args.append(f"-i {charging}={chunk_file}")

    return " ".join(args)
