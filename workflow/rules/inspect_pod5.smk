"""
POD5 inspection rules for detecting corrupted files.

Runs pod5 inspect on all unmerged POD5 files and generates a summary report.
"""

import os
from pathlib import Path


# ============================================================================
# Helper functions
# ============================================================================


def get_sample_pod5_files(wildcards):
    """Get all raw POD5 files for a specific sample."""
    sample_config = config["samples"][wildcards.sample]

    if "raw_pod5_list" in sample_config:
        return sample_config["raw_pod5_list"]
    elif "raw_pod5" in sample_config:
        raw_pod5 = sample_config["raw_pod5"]
        if os.path.isdir(raw_pod5):
            return list(Path(raw_pod5).rglob("*.pod5"))
        else:
            return [raw_pod5]
    else:
        raise ValueError(
            f"Sample {wildcards.sample} must have 'raw_pod5' or 'raw_pod5_list' in config"
        )


# ============================================================================
# Rules
# ============================================================================


rule inspect_sample_pod5s:
    """
    Inspect all POD5 files for a sample and generate a report.

    Runs pod5 inspect on each file and aggregates the results.
    """
    output:
        report="results/pod5_inspection/{sample}/inspection_report.tsv"
    params:
        pod5_files=get_sample_pod5_files
    threads: 4
    resources:
        mem_mb=4000,
        runtime=60
    log:
        "results/pod5_inspection/{sample}/inspect.log"
    run:
        import subprocess
        import os

        # Create output directory
        os.makedirs(os.path.dirname(output.report), exist_ok=True)

        # Open output file
        with open(output.report, "w") as out:
            # Write header
            out.write("sample\tfile\tstatus\tnum_reads\tfile_size_mb\tavg_read_length\tmin_read_length\tmax_read_length\terror_message\n")

            # Process each POD5 file
            for pod5_file in params.pod5_files:
                pod5_path = str(pod5_file)

                # Initialize values
                status = "OK"
                num_reads = "NA"
                file_size = "NA"
                avg_len = "NA"
                min_len = "NA"
                max_len = "NA"
                error_msg = ""

                # Get file size
                try:
                    file_size_bytes = os.path.getsize(pod5_path)
                    file_size = f"{file_size_bytes / (1024*1024):.2f}"
                except Exception as e:
                    status = "ERROR"
                    error_msg = f"Cannot access file: {str(e)}"
                    out.write(f"{wildcards.sample}\t{pod5_path}\t{status}\t{num_reads}\t{file_size}\t{avg_len}\t{min_len}\t{max_len}\t{error_msg}\n")
                    continue

                # Run pod5 inspect
                try:
                    result = subprocess.run(
                        ["pod5", "inspect", "summary", pod5_path],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode != 0:
                        status = "ERROR"
                        error_msg = f"pod5 inspect failed: {result.stderr[:200]}"
                    else:
                        # Parse output
                        output_text = result.stdout
                        for line in output_text.split('\n'):
                            if "Read count:" in line:
                                num_reads = line.split(":")[-1].strip()
                            elif "Mean:" in line and "NA" in avg_len:
                                # Look for mean read length
                                try:
                                    avg_len = line.split(":")[-1].strip().split()[0]
                                except:
                                    pass
                            elif "Min:" in line and "NA" in min_len:
                                try:
                                    min_len = line.split(":")[-1].strip().split()[0]
                                except:
                                    pass
                            elif "Max:" in line and "NA" in max_len:
                                try:
                                    max_len = line.split(":")[-1].strip().split()[0]
                                except:
                                    pass

                except subprocess.TimeoutExpired:
                    status = "ERROR"
                    error_msg = "pod5 inspect timed out (>60s)"
                except FileNotFoundError:
                    status = "ERROR"
                    error_msg = "pod5 command not found"
                except Exception as e:
                    status = "ERROR"
                    error_msg = f"Unexpected error: {str(e)}"

                # Write row
                out.write(f"{wildcards.sample}\t{pod5_path}\t{status}\t{num_reads}\t{file_size}\t{avg_len}\t{min_len}\t{max_len}\t{error_msg}\n")


rule aggregate_all_inspections:
    """
    Aggregate inspection results across all samples into a single master report.

    This provides an overview of all POD5 files in the project and highlights
    any corrupted or problematic files.
    """
    input:
        reports=expand(
            "results/pod5_inspection/{sample}/inspection_report.tsv",
            sample=config["samples"].keys()
        )
    output:
        master_report="results/pod5_inspection/pod5_inspection_master_report.tsv",
        summary="results/pod5_inspection/pod5_inspection_summary.txt"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5
    run:
        import pandas as pd

        # Combine all sample reports
        all_data = []
        for report_file in input.reports:
            df = pd.read_csv(report_file, sep='\t')
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)

        # Save master report
        combined.to_csv(output.master_report, sep='\t', index=False)

        # Generate summary statistics
        with open(output.summary, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("POD5 Inspection Summary Report\n")
            f.write("=" * 80 + "\n\n")

            total_files = len(combined)
            ok_files = len(combined[combined['status'] == 'OK'])
            error_files = len(combined[combined['status'] == 'ERROR'])

            f.write(f"Total POD5 files inspected: {total_files}\n")
            f.write(f"OK files: {ok_files} ({ok_files/total_files*100:.1f}%)\n")
            f.write(f"ERROR files: {error_files} ({error_files/total_files*100:.1f}%)\n\n")

            if error_files > 0:
                f.write("=" * 80 + "\n")
                f.write("CORRUPTED/ERROR FILES:\n")
                f.write("=" * 80 + "\n")
                error_df = combined[combined['status'] == 'ERROR']
                for idx, row in error_df.iterrows():
                    f.write(f"\nSample: {row['sample']}\n")
                    f.write(f"File: {row['file']}\n")
                    f.write(f"Error: {row['error_message']}\n")

            # Summary by sample
            f.write("\n" + "=" * 80 + "\n")
            f.write("Summary by Sample:\n")
            f.write("=" * 80 + "\n")
            sample_summary = combined.groupby('sample').agg({
                'file': 'count',
                'status': lambda x: (x == 'OK').sum()
            }).rename(columns={'file': 'total_files', 'status': 'ok_files'})
            sample_summary['error_files'] = sample_summary['total_files'] - sample_summary['ok_files']

            for sample_name, row in sample_summary.iterrows():
                f.write(f"\n{sample_name}:\n")
                f.write(f"  Total files: {row['total_files']}\n")
                f.write(f"  OK: {row['ok_files']}\n")
                f.write(f"  Errors: {row['error_files']}\n")

            # Read count statistics (for OK files only)
            ok_df = combined[combined['status'] == 'OK'].copy()
            if len(ok_df) > 0:
                ok_df['num_reads'] = pd.to_numeric(ok_df['num_reads'], errors='coerce')
                f.write("\n" + "=" * 80 + "\n")
                f.write("Read Count Statistics (OK files only):\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total reads across all files: {ok_df['num_reads'].sum():,.0f}\n")
                f.write(f"Mean reads per file: {ok_df['num_reads'].mean():,.0f}\n")
                f.write(f"Median reads per file: {ok_df['num_reads'].median():,.0f}\n")
                f.write(f"Min reads per file: {ok_df['num_reads'].min():,.0f}\n")
                f.write(f"Max reads per file: {ok_df['num_reads'].max():,.0f}\n")
