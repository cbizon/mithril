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

- **Rollup = query expansion.** The matrix for a general (class, predicate, class) is the
  boolean union of its descendants' matrices. For pair *existence*, the pair set of a general
  metapath is the boolean product of its unioned hop matrices, so no post-hoc aggregation
  is ever needed. The cost problem is the number of metapaths, not counting them.
- **Target-centric top-down search** (AMIE-style). Start from the most general metapath and
  specialize one position at a time (class -> subclass, predicate -> child predicate or
  qualified pseudo-predicate). Overlap with the target is anti-monotone under specialization,
  so min-support pruning is exact. Overlap is computed with target-masked products; a child's
  mask is its parent's overlap set. Open risk: the frontier may blow up faster than support
  pruning saves — measure before scaling.
- **Qualified predicates** become pseudo-predicates (e.g. affects + causes + increased +
  expression). They form a DAG (multiple parents), refined in a canonical order.
- **Simple (node-disjoint) paths** for <=3 hops are exact via inclusion-exclusion on
  count-semiring products (removes the target edge leaking into its own predictor):
  `simple = M1·M2·M3 − diag(M1·M2)·M3 − M1·diag(M2·M3) + M1∘M2ᵀ∘M3`, then drop a=d.
  Requires self-loops removed and one global node index.
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
mine     --target T --max-hops 3 -> rules + parent links                            (TODO)
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
