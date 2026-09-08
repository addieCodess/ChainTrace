# ChainTrace Evidence Report (clean_peel_chain)

**Unknown wallet:** `0xUNKNOWN_0099`

**Trail assessment: clean** -- no mixer/fan-out break detected in the traced path.

## Ranked attribution output

| VASP | Confidence |
|---|---|
| ExchangeA | 0.78 |

## Top node-level candidate

- Candidate deposit node: `0xDEP_ExchangeA_9900`
- Predicted VASP: **ExchangeA** (confidence 0.78)
- Heuristic agreement: True (score 0.667)
- Top contributing features (importance-weighted proxy, NOT SHAP):
  - `hop_distance_to_labeled` = 1.0 (weight 0.0)
  - `clustering_coeff` = 0.0 (weight 0.0)
  - `pagerank` = 0.0486 (weight 0.0)

## Grading (demo only -- would not appear in a real evidence pack)
- Ground truth VASP: ExchangeA
- Top prediction correct: **True**

*This output is a ranked probability with supporting evidence, not an assertion. Human investigator sign-off is required before any legal request is filed.*