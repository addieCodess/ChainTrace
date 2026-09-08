"""
build_graph.py

Turns the common edge-list schema -- (from, to, value, timestamp), the same
shape whether it came from ingestion/etherscan_client.py or
ingestion/synthetic_data.py -- into a NetworkX MultiDiGraph.

DECISION: NetworkX, not Neo4j, for this prototype.
The deck's own feasibility slide makes the case for us: a 2-3 hop
sub-graph of one address is thousands of nodes, not millions, and "runs
in-memory with NetworkX, scales to Neo4j." For a demo, standing up a Neo4j
instance adds real infra risk (a live demo dying on a DB connection is the
worst possible failure mode) for zero benefit at this graph size. Neo4j
earns its place in production once you need persistence across cases,
concurrent investigator queries, and Cypher-based multi-hop pattern
matching -- not for a single k-hop trace. See DECISIONS.md.
"""

from __future__ import annotations

import networkx as nx


def build_multidigraph(edges: list[tuple[str, str, float, int]]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for src, dst, value, ts in edges:
        g.add_edge(src, dst, value=value, timestamp=ts)
    return g


def merge_graphs(*graphs: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Combine the labeled-training-universe graph with the live
    investigation-case graph so hop-distance features can be computed
    against the full set of known clusters."""
    merged = nx.MultiDiGraph()
    for g in graphs:
        merged.add_edges_from(g.edges(data=True))
    return merged
