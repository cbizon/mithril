# BIOLINK-RULES

## Setup

- **uv** for package/environment management; the `.venv/` is sacred — always `uv run`.
  **NEVER install into system Python or the anaconda base environment.**
- **pytest** (functional style, doctests welcome) with high coverage. `uv run pytest`.
- TSV not CSV. Visualizations: PNG only, never PDF.
- Generated data goes under `data/` (git-ignored), e.g. `data/1.0.2/ingest/`.

## Purpose

Mine rules `N-hop metapath (N<=3) => 1-hop target edge type` from a Biolink KGX graph
(target: Translator KG 1.0.2, Biolink 4.4.4), with counts that follow **Biolink query
semantics**: a query for a predicate/class matches all of its descendants, and a
qualified predicate matches more specific qualifier values. Mining rules for a single
target edge type is the primary use case; "all targets" is a SLURM array over targets.

Replaces `../metapath-counts` and `../biolink-rule-mining` (and the June
`../kg-rule-mining` scaffold), which are archived. `../gandalf` is an external
dependency used only to verify final rules.

## Design

- **Rollup = query expansion.** A general hop label or class matches everything its
  descendants match, so the pairs joined by a general rule body are the *union* of the pairs
  joined by its specializations. Counts don't add up across specializations; pair sets do.
  Everything below is built on unions of pair sets.
- **One engine for every rule shape** (`provenance.py`, `counter.py`, `pathsearch.py`). A
  body has up to six positions: an optional *direct* hop between the endpoints, and a path
  of 0, 2 or 3 hops `(h1, c1, h2, c2, h3)`. Direct-only = 1-hop rule; path-only = discovery
  rule; both = conditioned rule. Shorter paths hold a "none" type in the unused positions,
  so the same tables, counter and search serve all of them. There are no matrices.
- **Path types.** Stored edge kinds matched by exactly the same hop labels are one *edge
  type*; category sets in exactly the same classes are one *node type* (no rule can tell
  them apart). A *path type* is the (edge, node, edge, node, edge) types of one concrete
  simple path. A body matches a path type when every position matches; it joins a pair when
  the pair has a matching path type (and, if it has a direct hop, a matching direct edge).
- **Overlap by set union (exact).** One pass lists each target pair's path types and direct
  edge types (`pair_types`) and stores, per path type, the bitset of target pairs having it.
  A body's overlap is the popcount of the OR of its path types' bitsets, rolled up one
  position at a time (classes -> h1 -> h2 -> h3) so bodies sharing a prefix share the work.
  A direct label just restricts the target pairs to those with such an edge. Cost grows with
  target pairs x path types per pair; a target with millions of pairs would need a sample.
- **Support by sketch union (estimated, ~3%).** Whole-graph pair sets are too large to
  store, so one pass over every subject-class node (`type_sketches`) keeps a HyperLogLog
  sketch (1,024 registers) per (direct edge type or none, path type) of the pairs having it;
  sketches merge by register-wise max, so they roll up exactly like the bitsets. The pass
  depends on the config (which defines the types) and the endpoint classes, not on the
  target. Chosen rules get exact support from one filtered graph walk each (`count_pairs`).
- **Target-centric top-down search** (AMIE-style). Start from the most general body and
  specialize one position at a time (class -> subclass, hop label -> child label), positions
  in a fixed order (direct label, classes, then h1, h2, h3) so each body is reached once.
  Overlap is anti-monotone under specialization, so min-support pruning is exact. Bodies
  that are written differently but match the same edges are reported separately; rule-set
  selection drops them as redundant.
- **Qualified predicates** become pseudo-predicates (e.g. affects + causes + increased +
  expression). They form a DAG (multiple parents).
- **Simple (node-disjoint) paths** only, by construction of the enumeration (a 3-hop path
  a-b-c-d has four distinct nodes), so the target edge cannot leak into its own predictor.
  The direct hop's labels exclude everything the target matches, for the same reason.
  Requires self-loops removed (done at build).
- **Production fidelity.** Rules must behave under standard Translator querying, so the
  predictor side uses every edge as-is (no provenance filtering, no canonicalization at
  build). Which stored atoms/orientations a query predicate matches is decided by Gandalf's
  own `PredicateExpander`/`QualifierExpander` (`semantics.py`), quirks included (e.g.
  `overlaps` does not match stored `part_of`). Node in class C iff C is in its KGX category
  list (KGX lists are ancestor-closed), same as Gandalf.
- **Target evidence** is configurable (`target_record_exclusions`); default drops text-mined
  and negated records. Predictions are kept (strong GO logical predictions).
- Excluded: `subclass_of` as a hop; node-level subclass expansion. Hub filtering is optional,
  off by default.

## Stages

```
ingest   KGX jsonl -> compact int arrays + edge-type / category-set tables  (once per release)
build    ingest + config/default.yaml -> deduplicated (subject, object) pairs per semantic
         atom (stored predicate + qualifier values), stored orientation, target-evidence flag
mine     scripts/mine.py, four steps per target (one run directory):
           types    target pairs -> path types + direct edge types     (minutes, per target)
           sketch   all pairs -> sketches, in stripes of subjects      (CPU-hours; per config +
                    endpoint classes, shared between targets by symlink)
           search   every rule above the recall floor: 2/3-hop paths, direct edges,
                    direct AND path; exact overlap, estimated support  (minutes)
           select   discovery + conditioned rule sets, exact support for chosen rules
verify   top rules -> Gandalf exact counts + intermediate-node concentration        (TODO)
rank     subsumption/redundancy filter + ranking                                    (TODO)
```

## Data notes (1.0.2)

- Source: `kgx-storage.ci.transltr.io/releases/translator_kg/1.0.2/`, extracted at
  `../translator_kg/1.0.2/`. ~30.5 M edges (10.6 M `subclass_of`), 1.85 M nodes.
- Node `category` lists are the full ancestor closure in **no particular order**; derive
  the most specific classes from Biolink, never from list position.
- Both `gene_associated_with_condition` and `condition_associated_with_gene` are stored
  (183,145 each). Not canonicalized: production expansion decides what matches.
- Edges carry `negated` (3.4 K, mostly GO); kept as predictors, excluded as target evidence.
- `/projects/sequence_analysis` is shared and can run near-full: `df -h` before big runs.

## Rules of the road

- Ask clarifying questions.
- Don't make classes just to group code.
- No bandaids — fix root causes. No try/except to hide failures; fail fast.
- Don't return made-up results if something fails. If you can't access a file, ask.
- Don't delete a test without explicit permission; don't weaken tests to make them pass.
- Don't duplicate functions — change the one. Keep directories clean.
- Check `git status` before commits. In commits/PRs, never mention co-authorship or the tool used.
