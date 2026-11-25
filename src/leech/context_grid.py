"""Parsing of grid-search context specifications (torch-free).

Lives apart from :mod:`leech.gridsearch`, which imports torch at module level,
because the Snakemake pre-flight in ``pipeline/workflow/rules/common.smk`` has
to read the context grid at parse time (see :mod:`leech.leakage`).
:mod:`leech.gridsearch` re-exports both functions.
"""


def parse_values(spec: str) -> list[int]:
    """Parse a value specification into a list of integers.

    Supports three formats:
    - Range: ``start:stop:step`` — expands to ``range(start, stop + 1, step)`` (inclusive stop)
    - Comma-separated: ``200,500,1000``
    - Single value: ``500``

    Args:
        spec: Value specification string.

    Returns:
        List of integer values.

    Raises:
        ValueError: If the specification is invalid.

    Examples:
        >>> parse_values("200:1000:200")
        [200, 400, 600, 800, 1000]
        >>> parse_values("200,500,1000")
        [200, 500, 1000]
        >>> parse_values("500")
        [500]
    """
    spec = spec.strip()

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Range spec must be start:stop:step, got {len(parts)} parts: '{spec}'"
            )
        try:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError(f"Range spec contains non-integer values: '{spec}'") from None
        if step <= 0:
            raise ValueError(f"Step must be positive, got {step}")
        if start > stop:
            raise ValueError(f"Start ({start}) must be <= stop ({stop})")
        return list(range(start, stop + 1, step))

    if "," in spec:
        return [int(x.strip()) for x in spec.split(",")]

    return [int(spec)]


def parse_context_grid(
    context_grid: str | None = None,
    left_contexts: str | None = None,
    right_contexts: str | None = None,
) -> tuple[list[int], list[int]]:
    """Parse context grid strings into integer lists.

    Supports range syntax (``start:stop:step``), comma-separated lists,
    and single values.

    Args:
        context_grid: Fallback context values when left/right not provided
            (e.g., "200,500,1000" or "200:1000:200")
        left_contexts: Override left contexts, or None to use context_grid
        right_contexts: Override right contexts, or None to use context_grid

    Returns:
        Tuple of (left_contexts_list, right_contexts_list)

    Raises:
        ValueError: If context_grid is None and either left_contexts or
            right_contexts is also None

    Examples:
        >>> parse_context_grid("200,500,1000")
        ([200, 500, 1000], [200, 500, 1000])
        >>> parse_context_grid("200:1000:200")
        ([200, 400, 600, 800, 1000], [200, 400, 600, 800, 1000])
        >>> parse_context_grid("200,500", left_contexts="100,200", right_contexts="300,400")
        ([100, 200], [300, 400])
        >>> parse_context_grid(left_contexts="100,200", right_contexts="300,400")
        ([100, 200], [300, 400])
    """
    if context_grid is None:
        if left_contexts is None or right_contexts is None:
            msg = (
                "--context-grid is required when --left-contexts or "
                "--right-contexts is not provided"
            )
            raise ValueError(msg)

    left_list = parse_values(left_contexts if left_contexts is not None else context_grid)
    right_list = parse_values(right_contexts if right_contexts is not None else context_grid)

    return left_list, right_list
