# ChainTrace Evidence Report (broken_trail (mixer fan-out))

**Unknown wallet:** `0xUNKNOWN_0099`

**Trail assessment: BROKEN** -- a mixer-style fan-out was detected within 2 hops. Confidence scores below are discounted accordingly; treat these as investigative leads requiring corroboration, not attribution.

## Ranked attribution output

| VASP | Confidence |
|---|---|
| ExchangeB_OTC | 0.33 |

## Top node-level candidate

- Candidate deposit node: `0xMIX_0990`
- Predicted VASP: **ExchangeB_OTC** (confidence 0.33)
- Heuristic agreement: False (score 0.333)
- Top contributing features (importance-weighted proxy, NOT SHAP):
  - `pagerank` = 0.0069 (weight 0.2893)
  - `dormancy_s` = 700.0 (weight 0.1086)
  - `fan_in_ratio` = 1.0 (weight 0.0941)

## Grading (demo only -- would not appear in a real evidence pack)
- Ground truth VASP: ExchangeA
- Top prediction correct: **False**

*This output is a ranked probability with supporting evidence, not an assertion. Human investigator sign-off is required before any legal request is filed.*