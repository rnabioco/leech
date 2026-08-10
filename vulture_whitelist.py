"""Vulture whitelist — suppress false positives for dead code detection.

These names are used dynamically (click commands, PyTorch internals,
entry points, dunder protocols) and cannot be detected by static analysis.
"""

# click / rich-click command decorators register functions dynamically
main  # noqa: vulture
data_group  # noqa: vulture
model_group  # noqa: vulture
eval_group  # noqa: vulture

# __getattr__ / __dir__ — PEP 562 lazy-import protocol (src/leech/__init__.py)
__getattr__  # noqa: vulture
__dir__  # noqa: vulture

# rich-click config attributes set on module-level object
TEXT_MARKUP  # noqa: vulture
SHOW_ARGUMENTS  # noqa: vulture
GROUP_ARGUMENTS_OPTIONS  # noqa: vulture
MAX_WIDTH  # noqa: vulture

# PyTorch DataLoader / training internals
benchmark  # noqa: vulture

# scikit-learn-compatible API
predict_proba  # noqa: vulture

# Entry point (pyproject.toml [project.scripts])
check_rust  # noqa: vulture

# Context manager protocol
__enter__  # noqa: vulture
__exit__  # noqa: vulture

# Public library API kept for consumers
get_sequence  # noqa: vulture
reference_names  # noqa: vulture
get_logger  # noqa: vulture

# Rust helpers imported only by the test suite (vulture scans src/ only)
_rs_test_process_read  # noqa: vulture
_rs_extract_levels  # noqa: vulture
_rs_rough_rescale_quantile  # noqa: vulture
