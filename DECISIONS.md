# ChainTrace prototype -- decision log

This documents what was actually built for the working prototype, every
deliberate simplification versus the pitch deck, why each one was made, and
what the upgrade path looks like. Nothing here is hidden from a judge who
asks a hard question -- if anything, this document IS the answer to most of
the hard questions.

## What actually runs, end to end

```
ingestion/synthetic_data.py  -->  graph/build_graph.py  -->  features/extract_features.py
                                                                      |
                                                                      v
                                                     embeddings/structural_embedding.py
                                                                      |
                                                                      v
                                pipeline/rule_based_heuristic.py --> model/train_classifier.py
                                                                      |
                                                                      v
                                                            pipeline/run_demo.py
                                                       (ranking, trail-break discount,
                                                        evidence report + graph PNG)
```

`python3 -m pipeline.run_demo` and `python3 -m pipeline.run_demo --broken-trail`
both run in under two seconds and produce a ranked VASP list, a per-node
explainability trace, an evidence graph PNG, and a Markdown/JSON evidence
report in `outputs/`. `python3 -m pipeline.validate` is a smoke test that
both scenarios behave the way the deck claims they should.

---

## Decision 1: Synthetic data generator instead of live Etherscan calls

**What we did:** `ingestion/synthetic_data.py` generates transaction graphs
with the same *shapes* the deck describes -- peel chains, mixer fan-out,
deposit-cluster fan-in, hot-wallet sweeps -- instead of calling the live
Etherscan API.

**Why:** Two independent reasons, either one would have been sufficient:
1. The sandbox this prototype was built in only allows network egress to a
   fixed domain allow-list (pypi, npm, github, etc.); `api.etherscan.io` is
   not reachable from it. A live call would fail closed regardless of API
   key.
2. Even without that constraint, a live demo that depends on a third-party
   API responding correctly, within rate limits, in front of judges, is a
   real failure risk for very little payoff -- the pipeline logic being
   demonstrated doesn't change based on where the edge list came from.

**What we kept real:** `ingestion/etherscan_client.py` is a genuine,
runnable reference implementation (BFS k-hop pull, rate-limit sleep,
degree-capped expansion for hub pruning) that a judge can read and verify
matches the deck's stated approach. It's not a stub -- it's the real
client, just not wired into the live demo path.

**Swap-in path:** Set `ETHERSCAN_API_KEY` and change one call in
`run_demo.py` from `generate_investigation_case()` to
`EtherscanClient().fetch_khop_subgraph(address, k=2)`. Both paths return the
same `(from, to, value, timestamp)` edge schema, so nothing downstream
changes.

**What we'd do differently with more time:** Record and replay one *actual*
Etherscan response for a real, already-public incident (see "Further
improvements" below) so the demo runs on real data without needing network
access at all -- best of both worlds.

---

## Decision 2: Etherscan/Elliptic++/SNAP-style labels instead of real ones

**What we did:** three synthetic "known VASP" clusters (`ExchangeA`,
`ExchangeB_OTC`, `ExchangeC`) with fabricated but realistically-shaped
deposit clusters and hot wallets, standing in for what the deck names as
real label sources (Etherscan Label Word Cloud, Elliptic++, SNAP Ethereum
exchange address sets).

**Why:** Building a real label set means either scraping/downloading actual
labeled-address datasets (Elliptic++ alone is a multi-hundred-MB download
outside this sandbox's reach) or hand-curating real, current exchange
hot-wallet addresses -- both real work items, not demo blockers. The
generator lets us control the exact fan-in/timing signatures needed to
prove the *classifier and feature-extraction logic* work correctly, which
is the thing actually being demonstrated.

**Further improvement:** swap in real Elliptic++ or SNAP address-label
data for training, and use one or two *documented, publicly reported*
historical hacks (a hack where researchers/news later confirmed which
exchange the funds were cashed out through) as the "unknown wallet" test
case -- this makes the demo's correctness independently checkable by a
judge, which is much stronger than a self-graded synthetic case.

---

## Decision 3: NetworkX in-memory graph instead of Neo4j

**What we did:** the entire prototype runs on a NetworkX `MultiDiGraph`,
no Neo4j instance.

**Why:** The deck's own feasibility slide argues this exact point: "a 2-3
hop sub-graph of one address is thousands of nodes, not millions... runs
in-memory with NetworkX, scales to Neo4j." Standing up a Neo4j instance for
a demo this size adds real infrastructure risk (a live demo dying on a DB
connection failure is close to the worst possible failure mode) for zero
functional benefit at this graph size.

**Where Neo4j actually earns its place:** persistence across investigator
sessions, concurrent multi-analyst queries, and Cypher-based multi-hop
*pattern matching* queries ("find all addresses within 3 hops that swept
into any labeled VASP in the last 30 days") -- none of which a single-case
demo needs. This is a genuine near-term production item, not a demo
shortcut we'd never revisit.

---

## Decision 4: Hand-engineered structural features instead of real node2vec

**What we did:** `embeddings/structural_embedding.py` explicitly does *not*
run node2vec. Instead, PageRank, clustering coefficient, and hop-distance
to a known cluster (computed in `features/extract_features.py`) serve as
the "embedding" columns fed to the classifier.

**Why:**
- Reference node2vec implementations depend on `gensim`, a dependency
  chain with real ABI/version-pinning fragility -- a bad risk to introduce
  right before a live demo.
- At the graph sizes involved here (tens of nodes per case), node2vec's
  main benefit -- learning latent structural roles beyond what you'd
  hand-engineer -- is marginal. PageRank ("is this a hub"), clustering
  coefficient ("is this densely interconnected"), and hop-distance ("is
  this near known infrastructure") already capture the three structural
  questions node2vec would mostly rediscover at this scale.

**This is a named simplification, not an equivalent method** -- don't
present hand-engineered features as node2vec output. The file header in
`structural_embedding.py` says this explicitly so nobody on the team
forgets it mid-demo.

**Swap-in path:** Neo4j GDS exposes node2vec as a single call
(`gds.node2vec.stream`). The classifier consumes a plain feature matrix, so
swapping this module's output for a real 128-d embedding requires changing
only this one file.

---

## Decision 5: RandomForest instead of RandomForest + XGBoost

**What we did:** only `RandomForestClassifier` (scikit-learn) is wired in,
despite the deck naming both RF and XGBoost.

**Why:** scikit-learn is already a dependency with zero additional install
risk; RandomForest gives `.feature_importances_` for free, which we lean on
for the explainability proxy (Decision 6); and on a training set this
small (a handful of labeled deposit clusters per VASP), a shallow forest
(`max_depth=6`, `min_samples_leaf=2` -- deliberately conservative) is less
likely to overfit than a boosted model tuned for a much larger label set.

**Further improvement:** once there's a real, larger labeled dataset,
swap in `xgboost.XGBClassifier` (drop-in interface) and compare Top-1/Top-3
accuracy against the current RF baseline before deciding which ships.

---

## Decision 6: feature-importance-weighted proxy instead of SHAP

**What we did:** `model/train_classifier.py`'s `rank_candidates()` computes
`importance * |z-score vs training mean|` per feature as a per-prediction
explanation, explicitly labeled everywhere in code and output as **"NOT
SHAP."**

**Why:** SHAP is the deck's stated tool and is the right long-term choice
-- it gives genuine per-prediction attribution (which is what should
actually go in a legal evidence pack), not just global feature importance.
It was left out of the working prototype because it's an extra dependency
with its own version sensitivities, and because the deviation-weighted
proxy is honest, cheap to compute, and good enough to demonstrate that the
*architecture* supports per-prediction explainability -- which is the claim
that matters for the demo.

**Further improvement:** `pip install shap`, wrap the trained RF in
`shap.TreeExplainer`, and replace the proxy in `rank_candidates()`. This is
a contained, well-scoped follow-up, not a redesign.

---

## Decision 7: rule-based heuristic runs first and filters candidates, not just "ships first"

**What we did:** originally, every structurally-reachable node within
range of the unknown wallet (hop wallets included) was handed straight to
the ML classifier. **This produced a real bug during prototyping:** hop
wallets with no deposit-cluster shape at all were being confidently (and
wrongly) classified as belonging to a VASP, because the classifier will
always pick *some* class. The fix was to make `rule_based_heuristic.py`'s
fan-in/sweep-regularity check run **first**, and only pass nodes that pass
it (or, failing that, fail open to the full candidate set rather than
silently returning nothing) into the classifier.

**Why this matters beyond the demo:** it's a direct, working
implementation of the deck's own build-order claim ("rule-based fan-in
heuristic ships first; the ML ranker is layered on top") -- and it turned
out to be functionally necessary, not just a nice-to-have milestone
ordering. A classifier that's never asked "which VASP is this hop wallet"
in the first place can't get that question wrong.

---

## Decision 8: hop-distance feature must exclude the query node itself

**What we did:** `_hop_distance_to_labeled()` computes distance to the
nearest *other* known-anchor node, explicitly excluding the node being
scored.

**Why this was a bug, not a design choice, and is documented as one:**
the first version returned distance 0 whenever the query node was itself
in the labeled set. Every labeled training example is (trivially) itself
labeled, so every training row got hop-distance = 0 with zero variance
across all three VASP classes -- a tree-based model correctly assigns a
constant feature zero importance, silently discarding a feature the deck
explicitly calls out as one of the five core behavioural signals. Excluding
self-matches restores real signal: a fresh, never-labeled deposit address
now correctly measures its distance to the nearest **known VASP hot
wallet**, which is the actual cold-start mechanism this whole system
depends on.

This is flagged here specifically because it's the kind of bug that
produces a plausible-looking but structurally wrong result (the pipeline
still runs, still ranks something) -- worth checking for in any follow-up
feature you add.

---

## Decision 9: trail-break detection is a blunt, explicit discount, not a learned one

**What we did:** `_detect_trail_break()` in `run_demo.py` looks for a
mixer-style fan-out (a node with several outgoing edges and little fan-in
of its own) within 2 hops of the unknown wallet, and if found, multiplies
every reported confidence by a fixed 0.4 discount -- rather than letting
the classifier's own probability stand.

**Why a hard-coded discount instead of teaching the model to lower
confidence itself:** the deck's own risk mitigation is explicit --
*"detect and flag the break, drop confidence honestly instead of
guessing."* A learned discount could in principle be defeated by a
mixer pattern the model hasn't seen; a hard rule can't be talked out of
applying the penalty. For a system whose output feeds legal requests, a
guaranteed-to-apply penalty is worth more than a marginally smarter one
that might not fire on a novel mixer shape.

**Further improvement:** replace the single fan-out threshold with a small
library of known mixer/bridge structural fingerprints (Tornado
Cash-style pool contracts, cross-chain bridge contract patterns) so the
break-detector recognizes *specific* known laundering primitives, not just
"a node with several outputs" -- which today could also (rarely) flag a
legitimate high-volume payment processor. This is exactly the deck's
listed roadmap item ("bridge heuristics on the roadmap").

---

## What's deliberately out of scope for this prototype (and why)

| Deck feature | Status here | Why it's out of scope now |
|---|---|---|
| Live Etherscan/Blockchair ingestion | Reference client only, not called live | Sandbox network restriction + demo reliability (Decision 1) |
| Neo4j graph store | NetworkX in-memory | Unneeded infra risk at this graph size (Decision 3) |
| node2vec / Neo4j GDS embeddings | Structural-feature proxy | Dependency risk + marginal benefit at this scale (Decision 4) |
| XGBoost | RandomForest only | Fewer dependencies, better fit for a small label set (Decision 5) |
| SHAP explainability | Importance-weighted proxy | Dependency risk; proxy demonstrates the same architecture (Decision 6) |
| FastAPI/Celery/PostgreSQL/Redis backend | Not built | The demo is a single-process script; a backend has nothing to serve yet without a UI or multi-user need |
| React + D3 investigator UI | Static PNG + Markdown/JSON evidence report | Same evidence-pack *content*, without the frontend build/hosting surface area |
| Bitcoin/TRON chain support | Ethereum-shaped data only | Feature schema is chain-agnostic by design, but no second chain's ingestion client was built |
| Analyst feedback loop (promote confirmed addresses) | Not built | Needs a persistence layer (Neo4j/Postgres) and a UI to act on, neither of which exist yet |
| Degree-capped BFS / hub pruning at production scale | Present only as a parameter (`max_neighbors_per_hop`) in the reference Etherscan client | Never exercised against a real hub-sized wallet in the demo |

None of these are "we don't know how" gaps -- they're scope cuts made so
the parts that ARE built actually run reliably end to end, which is worth
more in a live demo than a wider set of half-working pieces.

---

## Further improvements, roughly in the order we'd tackle them

1. **Real historical test case.** Replace the synthetic investigation case
   with one real, publicly-documented hack whose eventual cash-out exchange
   was independently confirmed by researchers or news coverage. This is the
   single highest-leverage change for demo credibility -- it turns "trust
   our synthetic ground truth" into "here's a real case, here's the
   independent confirmation, here's what our system said."
2. **Real label data.** Pull Elliptic++ and/or Etherscan's public label
   export for real known-exchange deposit clusters, instead of synthetic
   ones, so the classifier is trained on real behavioral distributions.
3. **Real SHAP.** Swap in `shap.TreeExplainer` per Decision 6 -- contained
   change, biggest jump in explainability credibility.
4. **Mixer/bridge fingerprint library.** Replace the single fan-out
   threshold in trail-break detection with known structural signatures for
   specific mixers and bridge contracts, reducing false positives on
   legitimate high-volume wallets.
5. **Confidence calibration.** Right now, `predict_proba` output is used
   directly as "confidence" -- RandomForest probabilities are not
   well-calibrated out of the box. Platt scaling or isotonic regression
   (`sklearn.calibration.CalibratedClassifierCV`) would make "0.79
   confidence" mean something statistically closer to "79% of the time
   this prediction is correct," which matters a great deal for a
   court-facing number.
6. **Real node2vec via Neo4j GDS**, once the graph store exists, to
   recover the representational power hand-engineered features leave on
   the table for larger, less obviously-structured clusters.
7. **XGBoost comparison**, once a larger real label set exists to justify
   the extra model complexity.
8. **Minimal investigator UI** (even a single-page Streamlit app before a
   full React/D3 build) so the evidence pack can be demoed as something an
   investigator clicks through, not just a generated Markdown file.
9. **Cross-chain support** (Bitcoin, TRON) using the same feature schema,
   starting with a second ingestion client mirroring
   `etherscan_client.py`'s interface.
10. **Analyst feedback loop**, once there's a persistence layer: let a
    human confirm or reject a prediction, and feed confirmed labels back
    into future training runs -- this is what actually keeps a VASP label
    set from going stale, per the deck's own risk table.
