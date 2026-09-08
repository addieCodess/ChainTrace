"""
structural_embedding.py

DECISION: this prototype does NOT run real node2vec (biased random walks
+ Skip-gram, as in the deck's ML layer), and that's deliberate.

Why we foregoing it here:
  - The reference node2vec implementations depend on gensim, which pulls
    a heavy, occasionally fragile dependency chain (numpy/scipy ABI
    pinning issues are common) -- a bad risk to introduce right before a
    live demo, for a graph that's only ~30-60 nodes in the hackathon
    scope.
  - At this scale, node2vec's main benefit (learning representations that
    capture higher-order structural roles beyond hand-engineered stats)
    is marginal: PageRank + clustering coefficient + hop-distance already
    capture "is this node a hub", "is this node densely interconnected",
    and "is this node near a known cluster" -- the three things node2vec
    would mostly rediscover on a graph this small.

What we do instead: treat extract_features.py's structural columns
(pagerank, clustering_coeff, hop_distance) AS the embedding, and
concatenate them with the behavioural features before classification.
This is functionally "features instead of embeddings", which is a real,
named simplification versus the deck (Neo4j GDS node2vec -> 128-d vector
per wallet), not an equivalent method -- flagged here and in
DECISIONS.md so nobody demos this as "we ran node2vec."

Swap-in path for production: Neo4j GDS exposes node2vec as a one-line
graph-projection call (`gds.node2vec.stream`). The classifier in
model/train_classifier.py takes a plain feature matrix, so swapping this
module's output for a real 128-d embedding requires no changes downstream
-- only this file changes.
"""

from __future__ import annotations

import numpy as np

from features.extract_features import FEATURE_NAMES

EMBEDDING_NOTE = (
    "Structural-feature proxy in use (pagerank + clustering_coeff + "
    "hop_distance_to_labeled from extract_features.py). Real node2vec "
    "embeddings are NOT computed in this prototype -- see "
    "embeddings/structural_embedding.py and DECISIONS.md."
)


def build_feature_matrix(feature_vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Stacks the per-node feature vectors from extract_node_features into
    a single (n_nodes, n_features) matrix ready for the classifier. This is
    the seam where a real embedding table would be concatenated in."""
    node_ids = list(feature_vectors.keys())
    matrix = np.vstack([feature_vectors[n] for n in node_ids])
    return node_ids, matrix


def describe_columns() -> list[str]:
    return list(FEATURE_NAMES)
