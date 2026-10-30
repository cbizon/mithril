# biolink-rules

Mine N-hop (N<=3) metapath rules from Biolink KGX graphs with Biolink query semantics.
See `CLAUDE.md` for design and stages.

```bash
uv run biolink-rules ingest --kg ../translator_kg/1.0.2 --out data/1.0.2/ingest
uv run pytest
```
