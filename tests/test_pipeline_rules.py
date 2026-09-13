"""Snakemake pipeline rules invoke real leech CLI options.

`pipeline/workflow/rules/*.smk` calls `uv run leech <group> <cmd> --flag ...`
in `shell:` blocks that Snakemake only ever renders as a string -- nothing
checks a flag name against the CLI it is calling. rnabioco/leech#266 found
several ways that drifts: `train.smk`/`compare_models.smk` built
`--config {input.grid_search}` inside a `params:` lambda (`model train` has
`--model-config`, not `--config`), `compare_models.smk` passed
`--max-epochs`/`--param-grid` straight in `shell:` (`model optimize` has
`--epochs`/`--context-grid`), and none of the four `model train`/
`model optimize` invocations across both files passed the `--motif` option
those commands mark `required=True`. All were silent until a real run hit
them, because CI never executes these rules (see the project's pipeline
dry-run memory note).

Flags show up in two places a rule can put them: literally in the `shell:`
block, or assembled into a string inside `params:` (as `--config` was, via an
f-string in a lambda) and referenced from `shell:` by placeholder. This
extracts both, per rule, and resolves each `--flag` against the actual click
`Command` it calls (imported from `leech.cli`) -- so a renamed or removed CLI
option fails this test instead of a job three stages into a real run.

Flags built by a *shared* helper function in common.smk (e.g.
`build_reference_fasta_arg()`) aren't visible here -- they don't appear as
literal text in the calling rule. That's a coverage gap for those specific
flags, not a false pass: every flag this test does see is checked for real.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = REPO_ROOT / "pipeline" / "workflow" / "rules"
SMK_FILES = sorted(RULES_DIR.glob("*.smk"))

# Top-level `leech` command groups that take a second word naming the
# subcommand (`leech data prepare`, `leech model train`, `leech eval test`).
# Anything else (`leech predict`) is a single top-level command.
GROUPS = {"data", "model", "eval"}

_RULE_HEADER = re.compile(r"^rule\s+(\w+)\s*:", re.MULTILINE)
_INVOCATION_START = re.compile(r"uv run leech\b")
_FLAG = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")
_BAREWORD = re.compile(r"[a-z][a-z0-9-]*")
_SHELL_BLOCK = re.compile(r'shell:\s*"""(.*?)"""', re.DOTALL)
_PARAMS_BLOCK = re.compile(r"\n {4}params:\n(.*?)(?=\n {4}(?:shell|run|script)\s*:|\Z)", re.DOTALL)
_QUOTED_STRING = re.compile(r'f?"([^"]*)"|f?\'([^\']*)\'')


def _iter_rule_blocks(text: str):
    """Yield (rule_name, block_text) for each top-level `rule NAME:` in a file."""
    headers = list(_RULE_HEADER.finditer(text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        yield m.group(1), text[start:end]


def _invocation_tokens(header: str):
    """Parse the words right after `uv run leech` into command tokens.

    ("data", "prepare") for a two-word group command, ("predict",) for a
    single top-level one.
    """
    first_match = _BAREWORD.match(header)
    if first_match is None:
        return None
    first = first_match.group(0)
    rest = header[first_match.end() :].lstrip()
    second_match = _BAREWORD.match(rest)
    if first in GROUPS and second_match is not None:
        return (first, second_match.group(0))
    return (first,)


def _shell_invocations(block_text: str):
    """Yield command tokens for each `uv run leech ...` call in a rule's `shell:`."""
    for shell_body in _SHELL_BLOCK.findall(block_text):
        for m in _INVOCATION_START.finditer(shell_body):
            header = shell_body[m.end() :].strip()
            tokens = _invocation_tokens(header)
            if tokens is not None:
                yield tokens


def _literal_flags(block_text: str) -> set[str]:
    """All `--flag`-shaped tokens found in a rule's `shell:` and `params:` text.

    `shell:` is scanned directly (most flags are typed there literally).
    `params:` is scanned only inside quoted string literals, which is where a
    flag built by an f-string/lambda (like the original `--config` bug) lives
    -- scanning the whole params text unfiltered would also pick up plain
    prose in nearby comments/docstrings.
    """
    flags: set[str] = set()
    for shell_body in _SHELL_BLOCK.findall(block_text):
        flags.update(_FLAG.findall(shell_body))
    params_match = _PARAMS_BLOCK.search(block_text)
    if params_match:
        for pair in _QUOTED_STRING.findall(params_match.group(1)):
            literal = pair[0] or pair[1]
            flags.update(_FLAG.findall(literal))
    return flags


def _iter_rule_invocations(text: str):
    """Yield (rule_name, command_tokens, flags) for each rule that calls leech.

    `flags` is every `--flag`-shaped token found anywhere in the rule (shell
    text plus quoted params strings), attributed to every `uv run leech ...`
    call the rule's `shell:` makes. Every rule in this pipeline calls at most
    one leech command per rule, so this attribution is exact today; a rule
    calling two different leech commands would get the same combined flag set
    checked against both (a conservative over-approximation, not a miss).
    """
    for rule_name, block_text in _iter_rule_blocks(text):
        tokens_list = list(_shell_invocations(block_text))
        if not tokens_list:
            continue
        flags = _literal_flags(block_text)
        for tokens in tokens_list:
            yield rule_name, tokens, flags


def _resolve_command(tokens):
    """Resolve ("data", "prepare") or ("predict",) to a click Command."""
    from leech.cli import cli

    first = tokens[0]
    if first in GROUPS:
        group = cli.commands[first]
        return group.commands[tokens[1]]
    return cli.commands[first]


def _declared_flags(command) -> set[str]:
    """Every `--flag` (including boolean secondary opts) a click Command declares."""
    flags: set[str] = set()
    for param in command.params:
        flags.update(opt for opt in getattr(param, "opts", []) or [] if opt.startswith("--"))
        flags.update(
            opt for opt in getattr(param, "secondary_opts", []) or [] if opt.startswith("--")
        )
    return flags


def _required_flags(command) -> set[str]:
    """The `--flag` form of every option a click Command marks `required=True`."""
    flags: set[str] = set()
    for param in command.params:
        if not getattr(param, "required", False):
            continue
        long_opts = [opt for opt in getattr(param, "opts", []) or [] if opt.startswith("--")]
        if long_opts:
            flags.add(long_opts[0])
    return flags


# Required options this static check cannot see supplied, because the calling
# rule builds them through a shared common.smk function that assembles a
# *different*, `multiple=True` flag per item rather than the long form once
# (mirrors the coverage gap already documented for build_*_arg() above).
# `data merge`'s `--input-chunks`/`-i` is built as `-i label=file -i label=file
# ...` by build_merge_input_args(); every merge_chunks*/merge_chunks_optimized*
# rule supplies it this way.
_REQUIRED_FLAG_EXCEPTIONS = {
    ("data", "merge"): {"--input-chunks"},
}


def test_there_are_smk_files_to_check():
    """Guards the glob: an empty list would make every test below vacuous."""
    assert SMK_FILES, f"no .smk files found under {RULES_DIR} -- has it moved?"


def test_there_are_leech_invocations_to_check():
    """Guards the extraction regex against silently matching nothing."""
    total = sum(1 for path in SMK_FILES for _r, _t, _f in _iter_rule_invocations(path.read_text()))
    assert total > 0, "found no `uv run leech ...` invocations in any .smk file"


@pytest.mark.parametrize("path", SMK_FILES, ids=lambda p: p.name)
def test_leech_invocations_resolve_to_real_commands(path: Path):
    """Every `leech <group> <cmd>` (or `leech <cmd>`) called must exist."""
    problems = []
    for rule_name, tokens, _flags in _iter_rule_invocations(path.read_text()):
        try:
            _resolve_command(tokens)
        except KeyError:
            problems.append(
                f"{path.name}:{rule_name}: no such leech command `leech {' '.join(tokens)}`"
            )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", SMK_FILES, ids=lambda p: p.name)
def test_leech_invocations_use_declared_flags(path: Path):
    """Every `--flag` passed to a `leech` invocation must be a real option.

    This is the regression guard for rnabioco/leech#266: it fails on
    `--config` (train.smk, compare_models.smk) and `--max-epochs`/
    `--param-grid` (compare_models.smk) before that issue's fix, and passes
    after it.
    """
    problems = []
    text = path.read_text()
    for rule_name, tokens, flags in _iter_rule_invocations(text):
        try:
            command = _resolve_command(tokens)
        except KeyError:
            # Reported by test_leech_invocations_resolve_to_real_commands.
            continue
        declared = _declared_flags(command)
        for flag in sorted(flags):
            if flag not in declared:
                problems.append(
                    f"{path.name}:{rule_name}: `leech {' '.join(tokens)}` has no "
                    f"{flag!r} option (declared: {sorted(declared)})"
                )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", SMK_FILES, ids=lambda p: p.name)
def test_leech_invocations_pass_required_flags(path: Path):
    """Every option a `leech` command marks `required=True` must actually be passed.

    A flag-validity check alone doesn't catch a missing flag, only a wrong
    one: rnabioco/leech#266 found `model train`/`model optimize` invocations
    that never passed `--motif`, required via the shared `model_provenance`
    decorator -- every one of those jobs failed at CLI argument parsing
    before running anything, and `test_leech_invocations_use_declared_flags`
    has nothing to say about an option that was never there to check.
    """
    problems = []
    text = path.read_text()
    for rule_name, tokens, flags in _iter_rule_invocations(text):
        try:
            command = _resolve_command(tokens)
        except KeyError:
            # Reported by test_leech_invocations_resolve_to_real_commands.
            continue
        exceptions = _REQUIRED_FLAG_EXCEPTIONS.get(tokens, set())
        missing = _required_flags(command) - flags - exceptions
        if missing:
            problems.append(
                f"{path.name}:{rule_name}: `leech {' '.join(tokens)}` is missing "
                f"required option(s) {sorted(missing)}"
            )
    assert not problems, "\n".join(problems)
