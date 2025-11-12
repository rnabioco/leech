# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting key design decisions made during the leech refactoring effort (January 2025).

## What are ADRs?

ADRs capture important architectural decisions along with their context and consequences. They help current and future developers understand:
- Why certain approaches were chosen
- What alternatives were considered
- What trade-offs were made

## Index

### Phase 1: Quick Wins

- [ADR 0001: Constants Consolidation](0001-constants-consolidation.md) - Single source of truth for defaults and magic numbers
- [ADR 0002: Structured Logging](0002-structured-logging.md) - Professional logging infrastructure replacing print() statements
- [ADR 0005: Sequence Encoding Consolidation](0005-sequence-encoding-consolidation.md) - Unified DNA sequence encoding

### Phase 2: High-Impact Refactorings

- [ADR 0003: Model Component Abstraction](0003-model-component-abstraction.md) - Reusable neural network branch components
- [ADR 0004: Inference Wrapper Pattern](0004-inference-wrapper-pattern.md) - Unified forward pass interface

## ADR Format

Each ADR follows this structure:

- **Status**: Accepted | Rejected | Deprecated | Superseded
- **Date**: When the decision was made
- **Context**: The problem or situation requiring a decision
- **Decision**: What was decided and how it works
- **Consequences**: Positive, negative, and neutral outcomes
- **Alternatives Considered**: Other options that were rejected

## Refactoring Impact

These ADRs document refactoring work that achieved:

- **Code reduction**: 500-600 lines eliminated (15-18% of codebase)
- **Maintainability**: Centralized configuration, logging, and component abstractions
- **Consistency**: Single source of truth for defaults and common patterns
- **Professionalism**: Structured logging, proper documentation

## Related Documentation

- [Architecture Overview](../architecture.md) - Overall architecture and module structure
- [Refactoring Guide](../refactoring_guide.md) - Migration guide and refactoring details
- [Documentation Home](../index.md) - Main documentation index

## Status Definitions

- **Accepted**: Decision has been implemented and is in use
- **Rejected**: Decision was considered but not implemented
- **Deprecated**: Decision was superseded by a later decision
- **Superseded**: Replaced by another ADR (link to replacement)

## Contributing

When adding new ADRs:

1. Use the next available number (0006, 0007, etc.)
2. Follow the standard format (see existing ADRs)
3. Update this README with a link
4. Use descriptive filenames (e.g., `0006-feature-name.md`)
