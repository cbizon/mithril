"""Exhaustive search for one target as a pool of short, resumable work units.

The coordinate-ordered search tree (``search.body_space``) splits into independent
subtrees. Each work unit is one subtree root (body, last coordinate, overlap); a worker
claims units, searches them with ``explore_ordered`` until its time budget runs out, and
writes any unexpanded frontier back to the pool as new units, so jobs stay short and
nothing is lost. ``feed`` keeps at most ``--max-jobs`` workers queued or running and
stops when the pool is drained.

    seed     create the pool: root expanded breadth-first into >= --units units
    work     (inside a SLURM job) claim and search units for --budget seconds
    status   counts and totals so far
    feed     keep the queue topped up to --max-jobs workers; requeue units of dead workers

Layout: <pool>/meta.json, pending/, claimed/<jobid>__<unit>.pkl, done/,
results/<unit>.tsv (passing bodies) and results/<unit>.json (counts).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import graphblas as gb

from biolink_rules.problem import load_problem
from biolink_rules.search import body_space, explore_ordered

DIRS = ("pending", "claimed", "done", "results")


def space_of(problem, meta: dict):
    return body_space(problem.store, problem.hops, problem.classes, problem.subject_class, problem.object_class,
                      meta["hops"], meta["cache_gb"], problem.contexts)


def open_problem(pool: Path):
    meta = json.loads((pool / "meta.json").read_text())
    problem = load_problem(Path(meta["data"]), Path(meta["config"]), meta["target"], meta["subject_class"],
                           meta["object_class"])
    assert problem.classes.labels == meta["class_labels"], "class lattice differs from the pool's"
    return meta, problem, space_of(problem, meta)


def write_unit(pool: Path, unit: str, body, last: int, mask: gb.Matrix, depth: int) -> None:
    rows, cols, _ = mask.to_coo()
    tmp = pool / "pending" / f".{unit}.tmp"
    with open(tmp, "wb") as handle:
        pickle.dump({"unit": unit, "body": body, "last": last, "depth": depth, "rows": rows, "cols": cols,
                     "shape": (mask.nrows, mask.ncols)}, handle)
    tmp.rename(pool / "pending" / f"{unit}.pkl")


def read_mask(unit: dict) -> gb.Matrix:
    return gb.Matrix.from_coo(unit["rows"], unit["cols"], True, nrows=unit["shape"][0], ncols=unit["shape"][1])


def write_result(pool: Path, unit: str, name, rows, evaluated: int, seconds: float, frontier: int) -> None:
    with open(pool / "results" / f"{unit}.tsv", "w") as handle:
        for body, depth, support in rows:
            handle.write(f"{name(body)}\t{support}\t{depth}\t{body!r}\n")
    (pool / "results" / f"{unit}.json").write_text(json.dumps(
        {"unit": unit, "evaluated": evaluated, "passed": len(rows), "seconds": seconds, "frontier": frontier,
         "job": os.environ.get("SLURM_JOB_ID", "")}) + "\n")


def seed(args) -> None:
    pool = args.pool
    if pool.exists():
        raise SystemExit(f"{pool} exists")
    for d in DIRS:
        (pool / d).mkdir(parents=True)
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    meta = {"data": str(args.data), "config": str(args.config), "target": args.target,
            "subject_class": args.subject_class, "object_class": args.object_class, "hops": args.hops,
            "recall_floor": args.recall_floor, "cache_gb": args.cache_gb, "target_pairs": problem.target.nvals,
            "min_support": int(args.recall_floor * problem.target.nvals), "class_labels": problem.classes.labels}
    (pool / "meta.json").write_text(json.dumps(meta))
    root, children, overlap, name = space_of(problem, meta)
    root_overlap = overlap(root, problem.target)
    write_result(pool, "seed-root", name, [(root, 0, root_overlap.nvals)], 1, 0.0, 0)
    # breadth-first: expand units one body at a time until there are enough
    queue = [("u", root, 0, root_overlap, 0)]
    count = 0
    while queue and len(queue) < args.units:
        unit, body, last, mask, depth = queue.pop(0)
        start = time.time()
        rows, evaluated, frontier = explore_ordered(children, overlap, body, last, mask, depth,
                                                    meta["min_support"], deadline=0.0)
        write_result(pool, f"seed{count}", name, rows, evaluated, time.time() - start, len(frontier))
        count += 1
        queue += [(f"{unit}.{i}", *entry) for i, entry in enumerate(frontier)]
        print(f"seed: expanded {unit} ({evaluated} evaluated), {len(queue)} units", flush=True)
    for unit, body, last, mask, depth in queue:
        write_unit(pool, unit, body, last, mask, depth)


def work_units(pool: Path, meta: dict, space, deadline: float, job: str) -> None:
    """Claim and search units until ``deadline`` (always at least one if any is pending)."""
    _, children, overlap, name = space
    while True:
        pending = sorted((pool / "pending").glob("*.pkl"))
        if not pending:
            return
        source = pending[0]
        claimed = pool / "claimed" / f"{job}__{source.name}"
        try:
            source.rename(claimed)
        except FileNotFoundError:  # another worker claimed it first
            continue
        with open(claimed, "rb") as handle:
            unit = pickle.load(handle)
        t0 = time.time()
        rows, evaluated, frontier = explore_ordered(children, overlap, unit["body"], unit["last"], read_mask(unit),
                                                    unit["depth"], meta["min_support"], deadline=deadline)
        for i, (body, last, mask, depth) in enumerate(frontier):
            write_unit(pool, f"{unit['unit']}.{i}", body, last, mask, depth)
        write_result(pool, unit["unit"], name, rows, evaluated, time.time() - t0, len(frontier))
        claimed.rename(pool / "done" / source.name)
        print(f"{unit['unit']}: {evaluated:,} evaluated, {len(rows):,} passed, {len(frontier)} handed back, "
              f"{time.time() - t0:.0f}s", flush=True)
        if time.time() >= deadline:
            return


def work(args) -> None:
    start = time.time()
    meta, _, space = open_problem(args.pool)
    print(f"setup {time.time() - start:.0f}s", flush=True)
    work_units(args.pool, meta, space, start + args.budget, os.environ["SLURM_JOB_ID"])


def totals(pool: Path) -> dict:
    stats = [json.loads(p.read_text()) for p in (pool / "results").glob("*.json")]
    return {"pending": len(list((pool / "pending").glob("*.pkl"))),
            "claimed": len(list((pool / "claimed").glob("*.pkl"))),
            "done": len(list((pool / "done").glob("*.pkl"))),
            "evaluated": sum(s["evaluated"] for s in stats), "passed": sum(s["passed"] for s in stats),
            "worker_seconds": sum(s["seconds"] for s in stats)}


def status(args) -> None:
    t = totals(args.pool)
    rate = t["evaluated"] / t["worker_seconds"] if t["worker_seconds"] else 0
    print(f"units pending {t['pending']:,} claimed {t['claimed']:,} done {t['done']:,}; evaluated "
          f"{t['evaluated']:,}, passed {t['passed']:,}; {t['worker_seconds'] / 3600:.1f} worker-hours "
          f"({rate:.1f} evaluations/s)")


def our_jobs(job_name: str) -> dict[str, str]:
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-n", job_name, "-o", "%i %T"],
                         check=True, capture_output=True, text=True).stdout
    return dict(line.split() for line in out.splitlines())


def feed(args) -> None:
    pool = args.pool.resolve()
    job_name = f"pool-{pool.name}"
    repo = Path(__file__).resolve().parent.parent
    log = pool / "workers"
    log.mkdir(exist_ok=True)
    while True:
        jobs = our_jobs(job_name)
        for claimed in (pool / "claimed").glob("*.pkl"):
            job, unit = claimed.name.split("__", 1)
            if job not in jobs:  # its worker died (time limit, node failure): put it back
                claimed.rename(pool / "pending" / unit)
                print(f"requeued {unit} from dead job {job}", flush=True)
        t = totals(pool)
        if t["pending"] == 0 and t["claimed"] == 0 and not jobs:
            print("pool drained", flush=True)
            status(args)
            return
        waiting = sum(state == "PENDING" for state in jobs.values())
        # top up, but never queue more workers than there are units waiting for them
        while len(jobs) < args.max_jobs and t["pending"] > waiting:
            out = subprocess.run(
                ["sbatch", "--parsable", f"--job-name={job_name}", "--partition", args.partition,
                 f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
                 f"--output={log}/%j.log", "--chdir", str(repo),
                 "--wrap", f"uv run python scripts/branch_pool.py work --pool {pool} --budget {args.budget}"],
                check=True, capture_output=True, text=True).stdout.strip()
            jobs[out] = "PENDING"
            waiting += 1
        print(f"[{time.strftime('%H:%M')}] jobs {len(jobs)} ({waiting} waiting); "
              f"units pending {t['pending']:,} claimed {t['claimed']:,} done {t['done']:,}; "
              f"evaluated {t['evaluated']:,}", flush=True)
        time.sleep(args.interval)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--pool", type=Path, required=True)
    s.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    s.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    s.add_argument("--target", required=True)
    s.add_argument("--subject-class", required=True)
    s.add_argument("--object-class", required=True)
    s.add_argument("--hops", type=int, default=3)
    s.add_argument("--recall-floor", type=float, default=0.02)
    s.add_argument("--cache-gb", type=float, default=16.0)
    s.add_argument("--units", type=int, default=64)
    w = sub.add_parser("work")
    w.add_argument("--pool", type=Path, required=True)
    w.add_argument("--budget", type=float, default=3000, help="seconds of searching before handing work back")
    st = sub.add_parser("status")
    st.add_argument("--pool", type=Path, required=True)
    f = sub.add_parser("feed")
    f.add_argument("--pool", type=Path, required=True)
    f.add_argument("--max-jobs", type=int, default=16, help="our jobs queued or running at once")
    f.add_argument("--budget", type=float, default=3000)
    f.add_argument("--time", default="01:15:00", help="SLURM limit: budget + setup + the last expansion")
    f.add_argument("--partition", default="batch")
    f.add_argument("--cpus", type=int, default=8)
    f.add_argument("--mem", default="64G")
    f.add_argument("--interval", type=float, default=60)
    args = p.parse_args()
    {"seed": seed, "work": work, "status": status, "feed": feed}[args.command](args)


if __name__ == "__main__":
    main()
