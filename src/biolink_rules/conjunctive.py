"""1-hop rules and 'direct edge AND path' rules.

The *direct* hop links the two endpoints themselves. Its lattice is the (subject class,
object class) hop context with every atom-orientation matched by the target removed,
so a rule can never use the edge it predicts (e.g. ``treats_or_applied_or_studied_to_treat``
becomes "trials / applied / studied, but not treats"). On its own a direct hop is a
1-hop rule; combined with a path it is a conjunctive rule (pairs satisfying both).
Overlap stays anti-monotone, so both walks prune on the same recall floor.
"""

from __future__ import annotations

import graphblas as gb

from biolink_rules.problem import Problem
from biolink_rules.score import Scored, Scorer


def direct_lattice(problem: Problem):
    return problem.contexts.lattice(problem.subject_class, problem.object_class, exclude=problem.target_bits)


def direct_label(problem: Problem, bits: int) -> str:
    return problem.contexts.label(bits, problem.subject_class, problem.object_class, exclude=problem.target_bits)


def walk_direct(problem: Problem, scorer: Scorer, base_overlap, floor: int):
    """Yield (direct bits, overlap pairs) for direct-lattice nodes whose overlap with
    ``base_overlap`` (a pair matrix) reaches ``floor``, top-down with pruning."""
    lattice, _ = direct_lattice(problem)
    if not lattice.sets:
        return
    seen, stack = set(), [lattice.root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        bits = lattice.sets[node]
        overlap = base_overlap.ewise_mult(scorer.direct(bits), gb.binary.pair).new()
        if overlap.nvals < floor:
            continue
        yield bits, overlap
        stack.extend(lattice.children[node])


def mine_one_hop(problem: Problem, scorer: Scorer, floor: int) -> list[Scored]:
    n = problem.target.nvals
    rules = []
    for bits, overlap in walk_direct(problem, scorer, problem.target, floor):
        body = (bits, ())
        support, capped = scorer.support(body, cap=0)
        rules.append(Scored(body, f"[{direct_label(problem, bits)}]", overlap.nvals, support, capped, n))
    return rules


def mine_conjunctions(problem: Problem, scorer: Scorer, paths: list[Scored], floor: int) -> list[Scored]:
    """Direct-AND-path rules for every path rule (already above ``floor``)."""
    n = problem.target.nvals
    rules = []
    for path_rule in paths:
        path = path_rule.body[1]
        path_overlap = scorer.overlap_pairs((0, path))
        for bits, overlap in walk_direct(problem, scorer, path_overlap, floor):
            body = (bits, path)
            support, capped = scorer.support(body, cap=0)
            name = f"[{direct_label(problem, bits)}] & {path_rule.name}"
            rules.append(Scored(body, name, overlap.nvals, support, capped, n))
    return rules
