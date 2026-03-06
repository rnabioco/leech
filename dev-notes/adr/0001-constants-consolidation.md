# ADR 0001: Constants Consolidation

**Status:** Accepted

**Date:** 2025-01-08

## Context

The leech codebase had magic numbers and default values scattered across multiple files:
- Model architectures hardcoded values like `signal_len=400`, `kmer_len=11`
- Training defaults like `batch_size=128`, `learning_rate=0.001` duplicated in CLI and training code
- Feature names and model hyperparameters repeated across modules
- No single source of truth for configuration values

This led to:
- Inconsistency when changing default values (updates needed in multiple locations)
- Difficulty understanding what values are configurable
- Risk of copy-paste errors when duplicating defaults
- Poor maintainability

## Decision

Create a centralized `src/leech/constants.py` module with all magic numbers and default values organized into logical groups:

1. **Signal processing defaults**: `DEFAULT_SIGNAL_CONTEXT`, `DEFAULT_KMER_CONTEXT`
2. **Model architecture defaults**: `DEFAULT_CONV_CHANNELS`, kernel sizes, LSTM hidden sizes
3. **Feature definitions**: `DWELL_FEATURES`, `SIGNAL_FEATURES` lists
4. **Training defaults**: `DEFAULT_BATCH_SIZE`, `DEFAULT_LEARNING_RATE`, `DEFAULT_EPOCHS`
5. **Model defaults**: `DEFAULT_SIGNAL_LEN`, `DEFAULT_KMER_LEN`, `DEFAULT_NUM_FEATURES`

All modules import from this central location rather than hardcoding values.

## Consequences

### Positive

- **Single source of truth**: Changing a default value only requires editing one location
- **Discoverability**: New contributors can easily find all configurable parameters
- **Consistency**: Impossible to have different default values in different modules
- **Documentation**: Constants are well-documented with inline comments
- **Maintainability**: Reduces code duplication (~50 hardcoded values eliminated)

### Negative

- **Import overhead**: All modules now depend on constants.py
- **Migration effort**: Required updating 10+ files to use constants (one-time cost)

### Neutral

- Constants are uppercase by Python convention (e.g., `DEFAULT_BATCH_SIZE`)
- Feature name lists provide clear documentation of feature engineering approach

## Alternatives Considered

1. **Configuration files (YAML/TOML)**: Rejected because constants are code-level defaults, not user configuration
2. **Pydantic models**: Deferred to avoid over-engineering for simple constants
3. **Dataclasses**: Unnecessary complexity for flat constant definitions

## Notes

This ADR implements Task 1 from the refactoring plan, eliminating ~50 magic numbers and improving maintainability.
