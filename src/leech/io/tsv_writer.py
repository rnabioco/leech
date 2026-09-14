"""TSV writer for prediction output.

Writes per-read predictions as gzipped (or plain) TSV instead of BAM tags.
Only multiclass models are supported.
"""

import gzip
import logging
from pathlib import Path

from leech.constants import BELOW_THRESHOLD_LABEL

logger = logging.getLogger("leech.io.tsv_writer")


def _unpack_pred(
    pred: tuple,
) -> tuple[int, int, float, list[float], float | None, int | None, bool]:
    """Unpack one ``pending[read_id]`` entry, any historical width.

    Mirrors ``leech.inference.helpers._unpack_multiclass_pred`` -- kept as its
    own copy rather than an import from ``inference`` into ``io``, which would
    invert this package's layering. 4-tuple (no CL head), 5-tuple (+ CL
    prediction), 7-tuple (current, + junction_indel/junction_mapped -- issue
    #282).
    """
    if len(pred) == 7:
        return pred
    if len(pred) == 5:
        base_idx, cls_idx, conf, all_probs, cl_pred = pred
        return base_idx, cls_idx, conf, all_probs, cl_pred, None, False
    base_idx, cls_idx, conf, all_probs = pred
    return base_idx, cls_idx, conf, all_probs, None, None, False


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
    min_margin : int
        Margin threshold in 0-255 uint8 space, used only when
        ``abstain_on_junction_indel`` is set.
    abstain_on_junction_indel : bool
        When True, ``predicted_aa`` is overwritten with
        :data:`~leech.constants.BELOW_THRESHOLD_LABEL` for rows whose motif
        junction is disrupted (unmapped, or a nonzero CIGAR indel) AND whose
        margin is below ``min_margin`` (issue #282). ``junction_indel``/
        ``junction_mapped`` are always written as their own columns
        regardless of this flag, so the rule can be re-applied offline from
        the raw TSV.
    """

    def __init__(
        self,
        output_path: Path,
        class_names: list[str],
        has_cl: bool,
        copy_tags: list[str] | None = None,
        min_margin: int = 0,
        abstain_on_junction_indel: bool = False,
    ) -> None:
        self.output_path = output_path
        self.class_names = class_names
        self.has_cl = has_cl
        self.copy_tags = copy_tags or []
        self.min_margin = min_margin
        self.abstain_on_junction_indel = abstain_on_junction_indel

        if str(output_path).endswith(".gz"):
            self._fh = gzip.open(output_path, "wt")
        else:
            self._fh = open(output_path, "w")  # noqa: SIM115

        # Write header
        prob_cols = [f"prob_{name}" for name in class_names]
        cols = ["read_name", "ref_name", *prob_cols, "predicted_aa", "confidence", "margin"]
        if has_cl:
            cols.append("predicted_cl")
        cols.extend(["junction_indel", "junction_mapped"])
        for tag in self.copy_tags:
            cols.append(f"tag_{tag}")
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
            _, cls_idx, conf, all_probs, cl_pred, junction_indel, junction_mapped = _unpack_pred(
                preds[0]
            )

            predicted_aa = int_to_label.get(cls_idx, str(cls_idx))
            ref_name = aln.reference_name or ""

            # Margin: difference between top two probabilities
            sorted_probs = sorted(all_probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

            if self.abstain_on_junction_indel:
                margin_uint8 = int(min(255, max(0, round(margin * 255))))
                disrupted = not junction_mapped or (
                    junction_indel is not None and junction_indel != 0
                )
                if disrupted and margin_uint8 < self.min_margin:
                    predicted_aa = BELOW_THRESHOLD_LABEL

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
            parts.append("" if junction_indel is None else str(junction_indel))
            parts.append(str(junction_mapped))

            for tag in self.copy_tags:
                if aln.has_tag(tag):
                    parts.append(str(aln.get_tag(tag)))
                else:
                    parts.append("")

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
