# ADR 0002: Structured Logging Infrastructure

**Status:** Accepted

**Date:** 2025-01-08

## Context

The leech codebase used `print()` statements for user feedback and debugging:
- 76 `print()` statements scattered across 7 modules
- No consistent formatting or log levels
- Impossible to filter or redirect output
- No timestamps or module context
- Cannot write to log files for long-running jobs
- Difficult to debug issues from user reports

Scientific computing on HPC clusters requires:
- Log files for batch jobs (SLURM, PBS)
- Filtering debug output without changing code
- Structured output for analysis
- Timestamped events for performance debugging

## Decision

Implement structured logging using Python's standard `logging` module:

1. **Created `logging_config.py`** with:
   - `setup_logging()`: Configures package-wide logging with console and optional file output
   - `get_logger()`: Returns module-specific loggers
   - Consistent format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

2. **Replaced all `print()` statements** with appropriate log levels:
   - `logger.info()`: User-facing progress messages (76 replacements)
   - `logger.debug()`: Detailed debugging information
   - `logger.warning()`: Non-fatal issues
   - `logger.exception()`: Error context in try/except blocks

3. **Module-specific loggers**: Each module gets its own logger (e.g., `leech.training`, `leech.inference`)

4. **CLI integration**: `cli.py` calls `setup_logging()` to configure output

## Consequences

### Positive

- **Professional output**: Timestamps, module names, and log levels
- **Configurable verbosity**: Set `--log-level DEBUG` without code changes
- **Log files**: Batch jobs can write to files for post-mortem analysis
- **Filtering**: Can filter by module (e.g., only show `leech.training` logs)
- **Exception handling**: `logger.exception()` includes stack traces
- **Maintainability**: Centralized logging configuration

### Negative

- **Migration effort**: Updated 7 modules with 76 `print()` → `logger` changes (one-time cost)
- **Import requirement**: All modules must import logger

### Neutral

- Uses Python stdlib `logging` (no new dependencies)
- Log format is customizable via `setup_logging(format_string=...)`

## Alternatives Considered

1. **Third-party logging (loguru, structlog)**: Rejected to avoid new dependencies
2. **Keep `print()` statements**: Rejected due to lack of flexibility and professionalism
3. **Per-module configuration**: Rejected in favor of centralized setup

## Examples

### Before
```python
print(f"Training epoch {epoch}/{epochs}")
print(f"Loss: {loss:.4f}")
```

### After
```python
logger = logging.getLogger("leech.training")
logger.info(f"Training epoch {epoch}/{epochs}")
logger.info(f"Loss: {loss:.4f}")
```

### Usage
```bash
# Normal output
leech train --train-data data.npz --model ConvLSTMDwell

# Debug output
leech train --train-data data.npz --model ConvLSTMDwell --log-level DEBUG

# Log to file
leech train --train-data data.npz --model ConvLSTMDwell --log-file train.log
```

## Notes

This ADR implements Task 4 from the refactoring plan, replacing 76 `print()` statements and enabling professional logging for production use.
