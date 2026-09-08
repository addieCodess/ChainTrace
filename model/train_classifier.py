"""
train_classifier.py

DECISION: RandomForestClassifier, not XGBoost, for the prototype.
Both are named in the deck ("Random Forest + XGBoost") as the candidate
model family; we picked RandomForest alone for the working prototype
because (a) it's in scikit-learn, already a dependency, so it adds zero
new packages/version-pinning risk before a demo, (b) it hands us
`.feature_importances_` for free as an explainability signal without
needing a separate library, and (c) on a training set this small
(a handful of labeled deposit clusters), a deep gradient-boosted model is
more likely to overfit than a modest forest is. XGBoost is a drop-in
swap (`from xgboost import XGBClassifier`) once there's a real labeled
dataset large enough to benefit from boosting -- flagged in
DECISIONS.md as a near-term upgrade, not a redesign.

DECISION: feature-importance-weighted contribution, not SHAP, for
explainability in this prototype.
SHAP (`pip install shap`) is the deck's stated tool and is the right
long-term choice -- it gives per-PREDICTION attribution, not just
per-MODEL importance, which is what actually gets attached to a legal
evidence pack. We didn't wire it in here because (a) it's an extra
dependency with its own version sensitivities we didn't want to risk
right before a demo, and (b) for a single ranked prediction, feature
importances weighted by each feature's deviation from the training-set
mean is a reasonable, honest proxy that's easy to compute for the demo.
This is explicitly a proxy, not SHAP -- do not present it as SHAP output.
Swap-in path noted in DECISIONS.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from embeddings.structural_embedding import build_feature_matrix
from features.extract_features import FEATURE_NAMES

UNKNOWN_LABEL = "NO_MATCH"


def train(labeled_feature_vectors: dict[str, np.ndarray], labels: dict[str, str]) -> RandomForestClassifier:
    node_ids, matrix = build_feature_matrix(labeled_feature_vectors)
    y = [labels[n] for n in node_ids]
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,          # shallow on purpose: small training set, avoid overfit
        min_samples_leaf=2,
        random_state=13,
        class_weight="balanced",
    )
    clf.fit(matrix, y)
    return clf


def rank_candidates(
    clf: RandomForestClassifier,
    candidate_feature_vectors: dict[str, np.ndarray],
    confidence_floor: float = 0.15,
) -> list[dict]:
    """Scores each candidate deposit-cluster node and returns a ranked list
    of {node, vasp, confidence, top_features} -- the shape that feeds
    straight into the evidence report."""
    node_ids, matrix = build_feature_matrix(candidate_feature_vectors)
    proba = clf.predict_proba(matrix)
    class_names = clf.classes_
    importances = clf.feature_importances_
    train_means = np.mean(matrix, axis=0)  # proxy baseline for "deviation-weighted" explanation

    results = []
    for i, node in enumerate(node_ids):
        top_class_idx = int(np.argmax(proba[i]))
        confidence = float(proba[i][top_class_idx])
        vasp = class_names[top_class_idx]

        # explainability proxy: importance * |z-score vs training mean|
        deviations = np.abs(matrix[i] - train_means) / (np.std(matrix, axis=0) + 1e-6)
        contribution = importances * deviations
        top_idx = np.argsort(contribution)[::-1][:3]
        top_features = [
            {"feature": FEATURE_NAMES[j], "value": round(float(matrix[i][j]), 4), "weight": round(float(contribution[j]), 4)}
            for j in top_idx
        ]

        results.append(
            {
                "node": node,
                "predicted_vasp": vasp if confidence >= confidence_floor else UNKNOWN_LABEL,
                "confidence": round(confidence, 4),
                "all_class_probs": {c: round(float(p), 4) for c, p in zip(class_names, proba[i])},
                "top_features": top_features,
            }
        )
    return sorted(results, key=lambda r: r["confidence"], reverse=True)
