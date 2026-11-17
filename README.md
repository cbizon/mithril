# mithril

**M**odel-**I**nformed **T**yped **H**orn **R**ule **I**nduction over **L**inks

Mithril mines rules from a [Biolink](https://biolink.github.io/biolink-model/) knowledge
graph. A rule says "if two nodes are joined by a path of this shape, they are likely to be
joined by this edge". For example, for the edge *chemical treats disease*:

```
treats> | biolink:PhenotypicFeature | treats< | biolink:Gene | treats>
```

reads "the chemical treats a phenotype, which is also treated by a gene, which treats the
disease". Mithril finds every such path shape of up to three hops that covers enough of the
known edges, scores each one, and picks a small non-redundant set.

The "model-informed" part is that rules are counted the way a Biolink query engine would
answer them. A rule that says `related_to` matches every edge whose predicate is a
descendant of `related_to`; a rule that says `biolink:Disease` matches every node of a
subclass of Disease; a qualified predicate matches more specific qualifier values. Rules
can therefore be general or specific in each position independently, and the counts for a
general rule are correct (they are not sums over its specific versions).

## Terms

| Term | Meaning |
|---|---|
| target | The edge type to predict, with the classes of its two ends, e.g. `treats>` from `biolink:ChemicalEntity` to `biolink:DiseaseOrPhenotypicFeature` |
| target pair | A (subject, object) pair that has the target edge in the graph |
| body | The "if" side of a rule: a path shape of 2 or 3 hops, a direct edge, or a direct edge AND a path |
| hop label | A predicate, optionally with qualifiers, plus a direction: `affects[aspect=abundance]>` |
| middle class | The class required of an intermediate node on the path; `(any)` means no requirement |
| overlap | Number of target pairs the body joins |
| support | Number of pairs the body joins anywhere in the graph, between nodes of the target's end classes |
| recall | overlap ÷ number of target pairs |
| precision | overlap ÷ support |

Precision counts a predicted pair as wrong unless the target edge is already in the graph,
so it understates how often a rule is right about edges the graph is missing.

## How it works

Earlier versions of this project computed one sparse matrix product per body. With
millions of bodies that took thousands of CPU-hours. Mithril instead walks the graph once
per number and answers every body by look-up.

1. **Inventory.** List the kinds of edge and the kinds of node in the graph, and group the
   ones that no rule can tell apart. This gives a few hundred *edge types* and a few dozen
   *node types*. A *path type* is the sequence of edge and node types along one concrete
   path.
2. **Overlap, exact.** Walk every 2- and 3-hop path between the target pairs, once, and
   record for each path type which target pairs have it (one bit per pair). A body's
   overlap is the number of pairs in the OR of the rows of the path types it matches. The
   OR is done one body position at a time and the intermediate tables are kept, so bodies
   that share a prefix share the work.
3. **Support, estimated to about 3%.** Walk every path between all nodes of the two end
   classes, once. There are too many pairs to record, so each path type keeps a small
   fixed-size summary (a HyperLogLog sketch) from which the number of distinct pairs can be
   estimated. Summaries combine the same way the bit rows do. This walk does not depend on
   the target, only on its end classes, so targets with the same end classes share it.
4. **Search.** Start from the most general body and make it more specific one position at a
   time. A more specific body can only join fewer pairs, so once a body falls below the
   recall floor nothing beneath it is tried. Support is looked up only for bodies that pass.
5. **Select.** Choose a small rule set greedily, and recount the support of the chosen rules
   exactly.

Paths never visit a node twice, so the target edge cannot appear inside its own predictor.

`CLAUDE.md` has the design in more detail.

## Setup

Requirements:

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- A Biolink graph in KGX format: a directory with `nodes.jsonl` and `edges.jsonl`.
  Development used Translator KG 1.0.2 (1.85 million nodes, 30.5 million edges).

```bash
uv sync
uv run pytest
```

Which stored edges a rule label matches follows the reasoner-transpiler rules used by
Translator query engines such as [Gandalf](https://github.com/ranking-agent/gandalf).
They are implemented in `src/mithril/semantics.py` on top of the Biolink Model Toolkit
(`bmt`).

## Usage

Prepare the graph once per release:

```bash
uv run mithril ingest --kg ../translator_kg/1.0.2 --out data/1.0.2/ingest
uv run mithril build  --ingest data/1.0.2/ingest --out data/1.0.2/build
```

Mine rules for one target. All four steps write into one run directory:

```bash
run=data/1.0.2/rules/treats

uv run python scripts/mine.py types --run $run --target 'treats>' \
    --subject-class biolink:ChemicalEntity --object-class biolink:DiseaseOrPhenotypicFeature
uv run python scripts/mine.py sketch --run $run     # or --stripe i --of n, as parallel jobs
uv run python scripts/mine.py search --run $run --recall-floor 0.01
uv run python scripts/mine.py select --run $run
```

| Step | What it does | Cost for `treats`, Translator KG 1.0.2 | Rerun when |
|---|---|---|---|
| `types` | Walks paths between the target pairs (step 2 above) | 9 minutes, one core | The target changes |
| `sketch` | Walks paths between all pairs (step 3) | CPU-hours; 8 stripes run side by side | The config or the end classes change |
| `search` | Scores every body above the recall floor | 55 minutes, one core, for 116.8 million bodies | Any time, e.g. a new floor |
| `select` | Chooses the rule set, exact support for chosen rules | About a minute | Any time |

The `types` and `search` steps were run with 64 to 128 GB of memory and each `sketch`
stripe with 48 GB. These are what was requested, not measured peaks.

### Output

In the run directory:

| File | Contents |
|---|---|
| `rules.tsv` | Every rule above the recall floor with estimated precision of at least `--min-precision` (default 0.01): `body`, `overlap`, `support_estimate`, `precision_estimate`, `recall` |
| `select_*.md` | The chosen rule sets, with exact support and precision |
| `mined.npz` | Every body that passed the floor, in compact form; `select` reads this |
| `types.npz`, `sketch*.npz` | The tables from the two graph walks |

`select` produces two sets that are chosen and judged separately:

- **Discovery** rules are paths only. They can predict a pair that has no edge between its
  ends. They are chosen for precision, subject to each rule adding new target pairs.
- **Conditioned** rules require a direct edge between the pair, alone or together with a
  path, for example `[studied_to_treat>] & treats> | biolink:PhenotypicFeature | ...`. They
  only rank pairs that already have a related edge, so their recall is not comparable to
  the discovery set.

### Reading a body

```
[studied_to_treat>] & affects[aspect=abundance]> | biolink:Gene | treats< | (any) | treats>
```

- `[...] &` at the start is a required direct edge between the two ends. Absent for
  discovery rules.
- Hop labels and middle classes alternate, separated by `|`: hop 1, middle class 1, hop 2,
  middle class 2, hop 3. A 2-hop body has one middle class.
- `>` means the edge points along the path (the node on the left is its subject); `<`
  means it points backwards.
- Qualifiers are in square brackets after the predicate.

## Configuration

`config/default.yaml` controls:

- which predicates may never be a hop (`subclass_of` by default);
- which edge records count as evidence for a target edge (text-mined and negated records
  are left out by default; they are still used on the predictor side);
- which qualifiers appear in hop labels and how finely they are split;
- which classes may be a middle class.

The predictor side always uses the graph as a query engine would see it. No edge is
removed from predictor paths.

## Results so far

One target has been run: `treats>` from ChemicalEntity to DiseaseOrPhenotypicFeature on
Translator KG 1.0.2 (58,464 target pairs), at a 1% recall floor.

- 34.9 million bodies passed the floor.
- The 20 chosen discovery rules together cover 29.9% of the target pairs, with mean
  precision 0.123.
- The 10 chosen conditioned rules have mean precision 0.506.

Most of the chosen discovery rules pass through other `treats` edges of the same chemical.
Whether to keep such edges out of rule bodies is an open question.

## Limits and open work

- Paths are at most three hops.
- The overlap step stores one bit per target pair per path type. A target with millions of
  pairs would need to be sampled; this is not implemented.
- Support in `rules.tsv` is an estimate. Only the chosen rules get exact support.
- Chosen rules have not yet been checked by running them through the query engine.
- Only one target has been run. Running all targets is a job array over targets that
  shares the `sketch` step between targets with the same end classes.

## Layout

```
src/mithril/
  ingest.py       KGX jsonl -> compact arrays
  build.py        ingest + config -> deduplicated edges per stored predicate and qualifiers
  semantics.py    which stored edges a query label matches
  lattice.py      the general-to-specific orderings of hop labels and classes
  contexts.py     hop labels that can occur between a given pair of classes
  store.py        the built graph, loaded
  problem.py      one target: its pairs, end classes and label orderings
  provenance.py   the graph walks: path types per pair, sketches, exact pair counts
  counter.py      overlap and support for a body from the walk tables
  pathsearch.py   the general-to-specific search
  score.py        rule-set selection
  report.py       rule tables
scripts/mine.py   the four mining steps
tests/            every count is checked against brute force on small graphs
```

## License

MIT. See `LICENSE`.
