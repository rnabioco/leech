"""Tests for shared CLI option helpers in leech.cli_options."""

import click
import pytest

from leech.cli_options import validate_selection_metric


class TestValidateSelectionMetric:
    """The --checkpoint-metric / --selection-metric click callback (#280).

    Both options share this one validator so a malformed parametric metric
    ("tpr_at_fpr:abc") is rejected at the CLI boundary with a clear
    click.BadParameter, rather than surfacing as a bare ValueError deep
    inside training or a grid search worker.
    """

    @pytest.mark.parametrize("value", ["auto", "val_acc", "val_f1", "val_auc"])
    def test_plain_names_pass_through(self, value):
        assert validate_selection_metric(None, None, value) == value

    @pytest.mark.parametrize(
        "value", ["tpr_at_fpr:0.0034", "callable_at_precision:0.99", "tpr_at_fpr:0"]
    )
    def test_valid_parametric_names_pass_through(self, value):
        assert validate_selection_metric(None, None, value) == value

    def test_unknown_kind_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            validate_selection_metric(None, None, "roc_auc:0.5")

    def test_non_numeric_parameter_raises_bad_parameter(self):
        with pytest.raises(click.BadParameter):
            validate_selection_metric(None, None, "tpr_at_fpr:not_a_number")

    def test_bare_unknown_name_is_not_validated_here(self):
        """A plain name with no colon is out of this function's scope --
        Trainer/gridsearch's own resolvers are what reject it, since this
        one callback serves both --checkpoint-metric (allows "auto") and
        --selection-metric (whose resolved value never is "auto")."""
        assert validate_selection_metric(None, None, "not_a_real_metric") == "not_a_real_metric"
