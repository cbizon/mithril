"""Markdown tables for rule lists and rule sets."""

from __future__ import annotations

from mithril.score import Scored


def rule_table(rows: list[Scored], set_columns: list[tuple[int, float]] | None = None) -> list[str]:
    """Markdown table; ``set_columns`` adds (new pairs, cumulative union recall) per row."""
    head = "| # | rule body | overlap | support | precision | recall | F1 |"
    if set_columns:
        head += " new | union recall |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, r in enumerate(rows, 1):
        line = (f"| {i} | `{r.name}` | {r.overlap:,} | {r.support:,} | {r.precision:.3f} | {r.recall:.3f} | "
                f"{r.f1:.3f} |")
        if set_columns:
            new, union = set_columns[i - 1]
            line += f" {new:,} | {union:.3f} |"
        lines.append(line)
    return lines
