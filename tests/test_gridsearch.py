"""Tests for grid search parsing utilities."""

import pytest

from leech.gridsearch import parse_context_grid, parse_values


class TestParseValues:
    """Tests for parse_values."""

    def test_range_syntax(self):
        assert parse_values("200:1000:200") == [200, 400, 600, 800, 1000]

    def test_range_syntax_inclusive_stop(self):
        """Stop value is included when step divides evenly."""
        assert parse_values("0:1000:500") == [0, 500, 1000]

    def test_range_syntax_stop_not_evenly_divisible(self):
        """Stop value is excluded when step doesn't divide evenly."""
        assert parse_values("200:1000:300") == [200, 500, 800]

    def test_range_syntax_single_step(self):
        assert parse_values("100:100:50") == [100]

    def test_comma_separated(self):
        assert parse_values("200,500,1000") == [200, 500, 1000]

    def test_comma_separated_with_spaces(self):
        assert parse_values("200, 500, 1000") == [200, 500, 1000]

    def test_single_value(self):
        assert parse_values("500") == [500]

    def test_single_value_zero(self):
        assert parse_values("0") == [0]

    def test_range_invalid_parts(self):
        with pytest.raises(ValueError, match="must be start:stop:step"):
            parse_values("100:200")

    def test_range_four_parts(self):
        with pytest.raises(ValueError, match="must be start:stop:step"):
            parse_values("100:200:50:10")

    def test_range_non_integer(self):
        with pytest.raises(ValueError, match="non-integer"):
            parse_values("100:abc:50")

    def test_range_zero_step(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            parse_values("100:200:0")

    def test_range_negative_step(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            parse_values("100:200:-50")

    def test_range_start_greater_than_stop(self):
        with pytest.raises(ValueError, match="must be <= stop"):
            parse_values("1000:200:100")

    def test_invalid_single_value(self):
        with pytest.raises(ValueError):
            parse_values("abc")

    def test_whitespace_stripped(self):
        assert parse_values("  500  ") == [500]


class TestParseContextGrid:
    """Tests for parse_context_grid."""

    def test_symmetric_comma(self):
        left, right = parse_context_grid("200,500,1000")
        assert left == [200, 500, 1000]
        assert right == [200, 500, 1000]

    def test_symmetric_range(self):
        left, right = parse_context_grid("200:1000:200")
        assert left == [200, 400, 600, 800, 1000]
        assert right == [200, 400, 600, 800, 1000]

    def test_overrides(self):
        left, right = parse_context_grid(
            "200,500", left_contexts="100:300:100", right_contexts="400,500"
        )
        assert left == [100, 200, 300]
        assert right == [400, 500]

    def test_left_override_only(self):
        left, right = parse_context_grid("200,500", left_contexts="100:300:100")
        assert left == [100, 200, 300]
        assert right == [200, 500]

    def test_right_override_only(self):
        left, right = parse_context_grid("200,500", right_contexts="100:300:100")
        assert left == [200, 500]
        assert right == [100, 200, 300]
