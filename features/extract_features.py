"""
extract_features.py

Computes the per-wallet "behavioural fingerprint" features named in the
pitch deck: fan-in ratio, sweep-timing regularity, value entropy, dormancy,
hop-distance to labelled clusters -- plus a couple of structural extras
(PageRank, clustering coefficient) that stand in for node2vec embeddings
in this prototype (see embeddings/structural_embedding.py for why).

These features are exactly what makes the system able to score a wallet
that has NEVER been seen before: none of them depend on a lookup table,
they depend only on the shape of activity around the node.
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np

FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "fan_in_ratio",
    "sweep_interval_std",
    "value_entropy",
    "dormancy_s",
    "pagerank",
    "clustering_coeff",
    "hop_distance_to_labeled",
]


def _value_entropy(values: list[float]) -> float:
    """Shannon entropy over the distribution of incoming transaction values,
    bucketed log-scale. A deposit address fed by many similarly-sized
    customer payments has different entropy than one fed by structured,
    threshold-avoiding amounts."""
    if not values:
        return 0.0
    buckets = Counter(round(math.log1p(v), 1) for v in values)
    total = sum(buckets.values())
    probs = [c / total for c in buckets.values()]
    return -sum(p * math.log2(p) for p in probs)


def _sweep_interval_std(out_timestamps: list[int]) -> float:
    """Std-dev of gaps between consecutive outgoing sweeps. Exchanges sweep
    deposit addresses on a near-fixed schedule, so a low std here is a
    strong "this feeds an exchange" signal; a peel-chain hop wallet with
    one-off irregular outflows has high/undefined std."""
    if len(out_timestamps) < 2:
        return -1.0  # sentinel: not enough data to judge regularity
    ts = sorted(out_timestamps)
    gaps = np.diff(ts)
    return float(np.std(gaps))


def _hop_distance_to_labeled(g: nx.MultiDiGraph, node: str, labeled_nodes: set[str]) -> int:
    """Distance to the nearest OTHER known-anchor node (a labeled deposit
    cluster or a VASP's public hot wallet). Deliberately excludes `node`
    itself from the target set: if we didn't, every labeled training node
    would trivially score distance 0 to itself, the feature would have zero
    variance across the whole training set, and a tree-based model would
    (correctly) assign it zero importance -- which is exactly the bug this
    prototype hit before this exclusion was added. See DECISIONS.md,
    'hop-distance self-matching'."""
    others = labeled_nodes - {node}
    if not others:
        return 99
    try:
        undirected = g.to_undirected(as_view=True)
        lengths = nx.single_source_shortest_path_length(undirected, node, cutoff=6)
        distances = [d for n, d in lengths.items() if n in others]
        return min(distances) if distances else 99
    except nx.NodeNotFound:
        return 99


def extract_node_features(
    g: nx.MultiDiGraph,
    nodes: list[str],
    labeled_nodes: set[str],
    pagerank: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Returns {node_id: feature_vector} for the requested nodes."""
    pagerank = pagerank or nx.pagerank(nx.DiGraph(g), alpha=0.85)
    simple_g = nx.Graph(g)  # for clustering coefficient (undirected, no multi-edges)

    out: dict[str, np.ndarray] = {}
    for node in nodes:
        in_edges = list(g.in_edges(node, data=True))
        out_edges = list(g.out_edges(node, data=True))
        in_degree = len(in_edges)
        out_degree = len(out_edges)
        in_values = [d["value"] for _, _, d in in_edges]
        out_timestamps = [d["timestamp"] for _, _, d in out_edges]

        vec = np.array(
            [
                in_degree,
                out_degree,
                (in_degree / out_degree) if out_degree else float(in_degree),  # fan_in_ratio
                _sweep_interval_std(out_timestamps),
                _value_entropy(in_values),
                (max(out_timestamps) - max((d["timestamp"] for _, _, d in in_edges), default=0))
                if in_edges and out_timestamps
                else 0,  # dormancy: gap between last inflow and last outflow
                pagerank.get(node, 0.0),
                nx.clustering(simple_g, node) if node in simple_g else 0.0,
                _hop_distance_to_labeled(g, node, labeled_nodes),
            ],
            dtype=float,
        )
        out[node] = vec
    return out
