"""Estimate the full support-pruned search size and run time for one target (Knuth probes).

    probe      run --probes random probes (seeded) and pickle them to --out
    summarize  pool probe pickles: estimated evaluations / passing bodies per depth with
               standard errors, projected single-job hours at the probes' own evaluation
               rate, and the largest middle-class partitions
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import time
from pathlib import Path

from biolink_rules.problem import load_problem
from biolink_rules.search import estimate_size


def probe(args) -> None:
    t0 = time.time()
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    min_support = int(args.recall_floor * problem.target.nvals)
    print(f"setup {time.time() - t0:.0f}s; target pairs {problem.target.nvals:,}; min support {min_support:,}",
          flush=True)
    def report(result):
        print(f"probe: depth {max(result['evaluated'])} estimate {sum(result['evaluated'].values()):.3g} "
              f"evaluations; {result['evaluations']} done in {result['seconds']:.0f}s", flush=True)

    probes = estimate_size(problem.store, problem.hops, problem.classes, problem.target, problem.subject_class,
                           problem.object_class, args.hops, min_support, probes=args.probes, seed=args.seed,
                           cache_gb=args.cache_gb, contexts=problem.contexts, time_limit=args.time_limit,
                           progress=report)
    meta = {"target_label": problem.target_label, "hops": args.hops, "min_support": min_support,
            "class_labels": problem.classes.labels}
    with open(args.out, "wb") as handle:
        pickle.dump({"meta": meta, "probes": probes}, handle)


def summarize(args) -> None:
    probes, meta = [], None
    for path in args.pickles:
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
        assert meta is None or loaded["meta"] == meta, f"{path} is from a different setup"
        meta = loaded["meta"]
        probes += loaded["probes"]
    n = len(probes)

    def mean_se(values):
        values = list(values)
        return statistics.mean(values), statistics.stdev(values) / n**0.5 if n > 1 else float("nan")

    labels = meta["class_labels"]
    lines = [f"# {meta['target_label']}: {meta['hops']}-hop search size, min support {meta['min_support']:,}",
             "", f"{n} probes.", "", "| depth | evaluated | passing |", "|---|---|---|"]
    depths = sorted({d for p in probes for d in p["evaluated"]})
    for d in depths:
        e, _ = mean_se(p["evaluated"].get(d, 0.0) for p in probes)
        q, _ = mean_se(p["passed"].get(d, 0.0) for p in probes)
        lines.append(f"| {d} | {e:.3g} | {q:.3g} |")
    total, total_se = mean_se(sum(p["evaluated"].values()) for p in probes)
    passing, passing_se = mean_se(sum(p["passed"].values()) for p in probes)
    rate = sum(p["evaluations"] for p in probes) / sum(p["seconds"] for p in probes)
    hours = total / rate / 3600
    lines += ["", f"Estimated evaluations {total:.3g} (SE {total_se:.2g}); passing bodies {passing:.3g} "
              f"(SE {passing_se:.2g}).",
              f"Probe evaluation rate {rate:.1f}/s -> {hours:.3g} single-job hours "
              f"({hours / 24:.3g} days).",
              "", "Largest middle-class partitions (share of estimated evaluations):", "",
              "| middle classes | evaluated | share |", "|---|---|---|"]
    by_classes: dict[tuple, float] = {}
    for p in probes:
        for key, value in p["by_classes"].items():
            by_classes[key] = by_classes.get(key, 0.0) + value / n
    for key, value in sorted(by_classes.items(), key=lambda kv: -kv[1])[:args.top]:
        names = " / ".join(min(labels[c], key=len) for c in key)
        lines.append(f"| {names} | {value:.3g} | {value / total:.1%} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    pr.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    pr.add_argument("--target", required=True, help="hop label, e.g. 'treats>'")
    pr.add_argument("--subject-class", required=True)
    pr.add_argument("--object-class", required=True)
    pr.add_argument("--hops", type=int, default=3)
    pr.add_argument("--recall-floor", type=float, default=0.02)
    pr.add_argument("--probes", type=int, default=100)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--time-limit", type=float, default=3 * 3600)
    pr.add_argument("--cache-gb", type=float, default=24.0)
    pr.add_argument("--out", type=Path, required=True)
    su = sub.add_parser("summarize")
    su.add_argument("pickles", type=Path, nargs="+")
    su.add_argument("--top", type=int, default=20)
    su.add_argument("--out", type=Path)
    args = p.parse_args()
    probe(args) if args.command == "probe" else summarize(args)


if __name__ == "__main__":
    main()
