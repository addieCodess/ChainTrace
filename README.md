# ChainTrace prototype -- SIH26182

A working, runnable slice of the ChainTrace VASP attribution engine
described in the pitch deck: unknown wallet in, ranked+explainable VASP
attribution out. See `DECISIONS.md` for the full reasoning behind every
design choice, what was simplified for this prototype, and the upgrade
path for each simplification.

## Requirements

Python 3.10+, `networkx`, `scikit-learn`, `matplotlib`, `numpy` (all
standard, no extra installs needed in most environments).
`requests` is only needed for `ingestion/etherscan_client.py` (the real,
not-run-by-default live ingestion path).

## Run the demo

```bash
# Clean peel-chain case -- should correctly attribute the true VASP
python3 -m pipeline.run_demo

# Broken-trail case -- a mixer-style fan-out is injected; confidence
# should visibly drop rather than confidently guess wrong
python3 -m pipeline.run_demo --broken-trail

# Smoke-test both scenarios' core claims in one shot
python3 -m pipeline.validate
```

Each `run_demo` invocation writes to `outputs/`:
- `evidence_report_<case>.json` -- full machine-readable ranking + detail
- `evidence_report_<case>.md` -- the human-readable evidence pack
- `evidence_graph_<case>.png` -- the traced subgraph, unknown wallet
  highlighted in orange, top candidate deposit node in blue, VASP hot
  wallet labeled

## Project layout

```
ingestion/
  etherscan_client.py       real live-pull client (reference impl, not called by the demo)
  synthetic_data.py         generates the demo's transaction graphs
graph/
  build_graph.py            edge-list -> NetworkX MultiDiGraph
features/
  extract_features.py       the "behavioural fingerprint": fan-in, sweep
                             regularity, value entropy, dormancy, hop-distance
embeddings/
  structural_embedding.py   structural-feature proxy standing in for node2vec
model/
  train_classifier.py       RandomForest classifier + explainability proxy
pipeline/
  rule_based_heuristic.py   fan-in heuristic that pre-filters candidates
  run_demo.py               end-to-end pipeline + evidence-pack writer
  validate.py               smoke test for both demo scenarios
outputs/                    generated evidence reports + graph images
DECISIONS.md                full design-decision log
```

## What this is and isn't

This is a scoped-down, honest prototype built to run reliably in a live
demo -- not the full FastAPI/Celery/Neo4j/React stack in the deck.
Every simplification is documented in `DECISIONS.md`, along with why it
was made and exactly what changes to move toward the full architecture.
Read that file before fielding "why doesn't it do X" questions.
