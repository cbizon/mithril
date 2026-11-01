"""The pool workflow (seed, short workers handing frontiers back) finds every passing body."""

import argparse
import importlib.util
from pathlib import Path

from conftest import make_world

from biolink_rules.search import explore_ordered

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/branch_pool.py"


def load_script():
    spec = importlib.util.spec_from_file_location("branch_pool", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pool_finds_every_passing_body(tmp_path):
    make_world(tmp_path)
    pool_script = load_script()
    pool = tmp_path / "pool"
    common = dict(data=tmp_path, config=tmp_path / "config.yaml", target="treats>",
                  subject_class="biolink:ChemicalEntity", object_class="biolink:DiseaseOrPhenotypicFeature")
    pool_script.seed(argparse.Namespace(pool=pool, hops=3, recall_floor=0.2, cache_gb=0.1, units=4, **common))
    meta, problem, space = pool_script.open_problem(pool)
    while list((pool / "pending").glob("*.pkl")):
        pool_script.work_units(pool, meta, space, deadline=0.0, job="1")  # one expansion, then hand back
    assert not list((pool / "claimed").glob("*"))
    found = set()
    for tsv in (pool / "results").glob("*.tsv"):
        found |= {eval(line.rstrip("\n").split("\t")[3]) for line in tsv.read_text().splitlines()}

    root, children, overlap, _ = space
    rows, _, frontier = explore_ordered(children, overlap, root, 0, overlap(root, problem.target), 0,
                                        meta["min_support"])
    assert not frontier
    assert found == {root} | {row[0] for row in rows}
