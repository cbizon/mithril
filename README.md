# biolink-rules

Mine N-hop (N<=3) metapath rules from Biolink KGX graphs with Biolink query semantics.
See `CLAUDE.md` for design and stages.

```bash
uv run biolink-rules ingest --kg ../translator_kg/1.0.2 --out data/1.0.2/ingest
uv run biolink-rules build --ingest data/1.0.2/ingest --out data/1.0.2/build

# rules for one target (see scripts/mine.py for the steps)
run=data/1.0.2/rules/treats
uv run python scripts/mine.py types --run $run --target 'treats>' \
    --subject-class biolink:ChemicalEntity --object-class biolink:DiseaseOrPhenotypicFeature
uv run python scripts/mine.py sketch --run $run            # or --stripe i --of n as parallel jobs
uv run python scripts/mine.py search --run $run
uv run python scripts/mine.py select --run $run

uv run pytest
```
