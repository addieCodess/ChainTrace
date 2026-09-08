"""
rule_based_heuristic.py

The deck is explicit about build order: "rule-based fan-in heuristic ships
first; the ML ranker is layered on top of the same pipeline." This module
is that first layer, kept in the demo (not just built and discarded) for
two real reasons documented in DECISIONS.md:

  1. Cold-start safety net -- if a candidate cluster's VASP has too few
     labeled training examples for the classifier to have learned it
     (a brand-new or rarely-seen exchange), the ML layer's confidence
     will legitimately be low. The heuristic still gives an investigator
     *something* actionable ("this address has fan-in 12:1 and sweeps on
     a 5-minute cadence -- that's exchange-shaped behaviour") rather than
     nothing.
  2. Sanity check -- if the ML layer and the heuristic strongly disagree
     on the same candidate node, that disagreement is itself worth
     surfacing to the investigator rather than silently trusting the ML
     score.
"""

from __future__ import annotations

FAN_IN_THRESHOLD = 4.0
SWEEP_REGULARITY_MAX_STD = 400.0  # seconds; below this counts as "regular"


def heuristic_score(feature_vector) -> dict:
    """feature_vector follows FEATURE_NAMES order from extract_features.py:
    [in_degree, out_degree, fan_in_ratio, sweep_interval_std, value_entropy,
     dormancy_s, pagerank, clustering_coeff, hop_distance_to_labeled]"""
    fan_in_ratio = feature_vector[2]
    sweep_std = feature_vector[3]
    hop_distance = feature_vector[8]

    is_high_fan_in = bool(fan_in_ratio >= FAN_IN_THRESHOLD)
    is_regular_sweep = bool(0 <= sweep_std <= SWEEP_REGULARITY_MAX_STD)
    is_near_known_cluster = bool(hop_distance <= 2)

    signal_count = sum([is_high_fan_in, is_regular_sweep, is_near_known_cluster])
    score = signal_count / 3.0

    return {
        "heuristic_score": round(score, 3),
        "looks_like_deposit_cluster": bool(signal_count >= 2),
        "signals": {
            "high_fan_in": is_high_fan_in,
            "regular_sweep_timing": is_regular_sweep,
            "near_known_vasp_cluster": is_near_known_cluster,
        },
    }
