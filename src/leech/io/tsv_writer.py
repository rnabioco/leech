"""TSV writer for prediction output.

Writes per-read predictions as gzipped (or plain) TSV instead of BAM tags.
Only multiclass models are supported.
"""

import gzip
import logging
from pathlib import Path

logger = logging.getLogger("leech.io.tsv_writer")


class TsvPredictionWriter:
    """Write multiclass predictions to a TSV file.

    Parameters
    ----------
    output_path : Path
        Output file path. Gzip-compressed if ends with ``.gz``.
    class_names : list[str]
        Ordered class labels (matching model output indices).
    has_cl : bool
        Whether a CL regression head is present.
    """

    def __init__(self, output_path: Path, class_names: list[str], has_cl: bool) -> None:
        self.output_path = output_path
        self.class_names = class_names
        self.has_cl = has_cl

        if str(output_path).endswith(".gz"):
            self._fh = gzip.open(output_path, "wt")
        else:
            self._fh = open(output_path, "w")  # noqa: SIM115

        # Write header
        prob_cols = [f"prob_{name}" for name in class_names]
        cols = ["read_name", "ref_name", *prob_cols, "predicted_aa", "confidence", "margin"]
        if has_cl:
            cols.append("predicted_cl")
        self._fh.write("\t".join(cols) + "\n")

    def write_predictions(
        self,
        aln_batch: list,
        pending: dict[str, list],
        int_to_label: dict[int, str],
    ) -> int:
        """Write predictions for a mega-batch of alignments.

        Parameters
        ----------
        aln_batch : list[pysam.AlignedSegment]
            Alignments from BAM (used for read_name and ref_name).
        pending : dict
            ``{read_id: [(base_idx, cls_idx, conf, probs, ...), ...]}``
        int_to_label : dict
            Index-to-label mapping.

        Returns
        -------
        int
            Number of reads written.
        """
        n_written = 0
        for aln in aln_batch:
            preds = pending.get(aln.query_name)
            if not preds:
                continue
            pred = preds[0]
            # Unpack: 4-tuple (old) or 5-tuple (with CL prediction)
            if len(pred) == 5:
                _, cls_idx, conf, all_probs, cl_pred = pred
            else:
                _, cls_idx, conf, all_probs = pred
                cl_pred = None

            predicted_aa = int_to_label.get(cls_idx, str(cls_idx))
            ref_name = aln.reference_name or ""

            # Margin: difference between top two probabilities
            sorted_probs = sorted(all_probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

            parts = [
                aln.query_name,
                ref_name,
                *[f"{p:.6f}" for p in all_probs],
                predicted_aa,
                f"{conf:.6f}",
                f"{margin:.6f}",
            ]
            if self.has_cl:
                parts.append(f"{cl_pred:.6f}" if cl_pred is not None else "")

            self._fh.write("\t".join(parts) + "\n")
            n_written += 1
        return n_written

    def close(self) -> None:
        """Flush and close the output file."""
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
