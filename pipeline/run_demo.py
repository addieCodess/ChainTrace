"""
run_demo.py

The single entry point that reproduces, end to end, everything the pitch
deck's "Idea Title" slide diagrams:

    unknown wallet -> k-hop graph -> behavioural fingerprint
    -> (structural embedding proxy) -> VASP classifier + heuristic
    -> ranked, explainable, exportable evidence pack

Run with:  python3 -m pipeline.run_demo [--broken-trail]

`--broken-trail` reruns the same case through a mixer-style fan-out
instead of a clean peel chain, to demonstrate the "confidence should drop
honestly, not guess" behaviour called out in the deck's risk table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extract_features import extract_node_features
from graph.build_graph import build_multidigraph, merge_graphs
from ingestion.synthetic_data import generate_investigation_case, generate_training_universe
from model.train_classifier import rank_candidates, train
from pipeline.rule_based_heuristic import heuristic_score

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _detect_trail_break(g: nx.MultiDiGraph, root: str, fan_out_threshold: int = 4) -> bool:
    """Looks for a mixer-style fan-out within 2 hops of the root: a node
    with several outgoing edges and little-to-no incoming fan-in of its
    own (a pass-through splitter, not a deposit address). This is the
    signal the deck's risk table calls for: 'Detect and flag the break,
    drop confidence honestly instead of guessing.' It is a coarse,
    single-signal heuristic -- a real deployment would maintain a
    fingerprint library for known mixer/bridge contracts (see
    DECISIONS.md, 'mixer/bridge detection')."""
    lengths = nx.single_source_shortest_path_length(g.to_undirected(as_view=True), root, cutoff=2)
    for node, dist in lengths.items():
        if node == root or dist == 0:
            continue
        if g.out_degree(node) >= fan_out_threshold and g.in_degree(node) <= 1:
            return True
    return False


def _candidate_deposit_nodes(g: nx.MultiDiGraph, unknown_wallet: str, max_hops: int = 3) -> list[str]:
    """Candidates = nodes within max_hops of the unknown wallet that have
    at least one outgoing edge (i.e. could plausibly be a deposit address
    sweeping onward) and are not the unknown wallet itself."""
    lengths = nx.single_source_shortest_path_length(g.to_undirected(as_view=True), unknown_wallet, cutoff=max_hops)
    return [n for n, d in lengths.items() if n != unknown_wallet and g.out_degree(n) > 0 and d > 0]


def run(broken_trail: bool = False) -> dict:
    training_cases = generate_training_universe()
    investigation_case = generate_investigation_case(broken_trail=broken_trail)

    training_graph = merge_graphs(*(build_multidigraph(c.edges) for c in training_cases))
    live_graph = build_multidigraph(investigation_case.edges)
    full_graph = merge_graphs(training_graph, live_graph)

    # `label_map` (deposit-cluster -> VASP) is what the classifier trains on.
    # `known_anchor_nodes` (deposit clusters + hot wallets) is the broader set
    # used for the hop_distance_to_labeled feature -- in reality, a VASP's
    # hot wallet is public/well-known even when today's deposit address
    # isn't, and that adjacency is exactly the cold-start signal this system
    # relies on. See _canonical_hot_wallet() in ingestion/synthetic_data.py.
    label_map: dict[str, str] = {}
    known_anchor_nodes: set[str] = set()
    for case in training_cases:
        known_anchor_nodes.add(case.hot_wallet)
        for dep in case.deposit_cluster:
            label_map[dep] = case.label
            known_anchor_nodes.add(dep)

    training_feature_vectors = extract_node_features(full_graph, list(label_map.keys()), known_anchor_nodes)
    clf = train(training_feature_vectors, label_map)

    all_candidates = _candidate_deposit_nodes(full_graph, investigation_case.unknown_wallet)
    all_candidate_vectors = extract_node_features(full_graph, all_candidates, known_anchor_nodes)

    # Rule-based layer runs FIRST and narrows the field, matching the deck's
    # stated build order ("rule-based fan-in heuristic ships first; the ML
    # ranker is layered on top of the same pipeline"). Practically, this also
    # keeps the classifier from ever being asked to name a VASP for a plain
    # pass-through hop wallet -- that's a job for structure, not for the
    # VASP-identity model, and conflating the two was an early bug in this
    # prototype (see DECISIONS.md, "candidate pre-filtering").
    heuristics = {node: heuristic_score(vec) for node, vec in all_candidate_vectors.items()}
    deposit_candidates = [n for n in all_candidates if heuristics[n]["looks_like_deposit_cluster"]]
    if not deposit_candidates:
        deposit_candidates = all_candidates  # fail open: let the ML layer see everything rather than nothing

    candidate_vectors = {n: all_candidate_vectors[n] for n in deposit_candidates}
    ml_ranking = rank_candidates(clf, candidate_vectors)
    for r in ml_ranking:
        r["heuristic"] = heuristics[r["node"]]

    # Trail-break handling: this is the concrete implementation of the
    # deck's risk-mitigation promise. A mixer-style fan-out means the
    # structural link between the unknown wallet and any downstream
    # deposit cluster is genuinely weaker evidence than a clean peel chain
    # -- so every confidence figure gets discounted, and no candidate is
    # allowed to look "confident" in the output regardless of what the
    # raw classifier probability said. This discount is a blunt, explicit
    # penalty rather than a learned one on purpose: it must be impossible
    # for the model to talk itself back up to false confidence.
    trail_broken = _detect_trail_break(full_graph, investigation_case.unknown_wallet)
    CONFIDENCE_DISCOUNT = 0.4
    if trail_broken:
        for r in ml_ranking:
            r["raw_confidence_before_discount"] = r["confidence"]
            r["confidence"] = round(r["confidence"] * CONFIDENCE_DISCOUNT, 4)

    # aggregate node-level candidate scores up to a VASP-level ranked list --
    # this is the "Ranked Attribution Output" table from the deck
    vasp_scores: dict[str, float] = {}
    for r in ml_ranking:
        vasp_scores[r["predicted_vasp"]] = max(vasp_scores.get(r["predicted_vasp"], 0.0), r["confidence"])
    vasp_ranking = sorted(vasp_scores.items(), key=lambda kv: kv[1], reverse=True)

    ground_truth_vasp = "ExchangeA"  # known only for grading the demo, never shown to the classifier
    top_prediction = vasp_ranking[0][0] if vasp_ranking else "NO_MATCH"
    correct = top_prediction == ground_truth_vasp

    report = {
        "case": "broken_trail (mixer fan-out)" if broken_trail else "clean_peel_chain",
        "unknown_wallet": investigation_case.unknown_wallet,
        "trail_break_detected": trail_broken,
        "ranked_vasps": [{"vasp": v, "confidence": c} for v, c in vasp_ranking],
        "node_level_detail": ml_ranking,
        "ground_truth_vasp_for_grading": ground_truth_vasp,
        "top_prediction_correct": correct,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = OUTPUT_DIR / f"evidence_report_{'broken' if broken_trail else 'clean'}.json"
    report_path.write_text(json.dumps(report, indent=2))

    _render_evidence_graph(full_graph, investigation_case, ml_ranking, broken_trail)
    _write_evidence_markdown(report, broken_trail)

    return report


def _render_evidence_graph(g, case, ml_ranking, broken_trail) -> None:
    subgraph_nodes = set(nx.single_source_shortest_path_length(g.to_undirected(as_view=True), case.unknown_wallet, cutoff=4))
    sub = g.subgraph(subgraph_nodes)

    top_candidate = ml_ranking[0]["node"] if ml_ranking else None
    pos = nx.spring_layout(sub, seed=7, k=0.9)

    node_colors = []
    for n in sub.nodes():
        if n == case.unknown_wallet:
            node_colors.append("#e07a1f")
        elif n == top_candidate:
            node_colors.append("#1f6fe0")
        elif n == case.hot_wallet:
            node_colors.append("#1f3a6e")
        else:
            node_colors.append("#9aa0a6")

    plt.figure(figsize=(9, 6))
    nx.draw_networkx_edges(sub, pos, alpha=0.4, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=260)
    labels = {n: ("UNKNOWN" if n == case.unknown_wallet else ("HOT-WALLET" if n == case.hot_wallet else "")) for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=7)
    plt.title(f"ChainTrace evidence graph -- {'broken trail' if broken_trail else 'clean peel chain'}")
    plt.axis("off")
    out_path = OUTPUT_DIR / f"evidence_graph_{'broken' if broken_trail else 'clean'}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _write_evidence_markdown(report: dict, broken_trail: bool) -> None:
    lines = [
        f"# ChainTrace Evidence Report ({report['case']})",
        "",
        f"**Unknown wallet:** `{report['unknown_wallet']}`",
        "",
        (
            "**Trail assessment: BROKEN** -- a mixer-style fan-out was detected within "
            "2 hops. Confidence scores below are discounted accordingly; treat these as "
            "investigative leads requiring corroboration, not attribution."
            if report["trail_break_detected"]
            else "**Trail assessment: clean** -- no mixer/fan-out break detected in the traced path."
        ),
        "",
        "## Ranked attribution output",
        "",
        "| VASP | Confidence |",
        "|---|---|",
    ]
    for r in report["ranked_vasps"]:
        lines.append(f"| {r['vasp']} | {r['confidence']:.2f} |")

    lines += [
        "",
        "## Top node-level candidate",
        "",
    ]
    if report["node_level_detail"]:
        top = report["node_level_detail"][0]
        lines.append(f"- Candidate deposit node: `{top['node']}`")
        lines.append(f"- Predicted VASP: **{top['predicted_vasp']}** (confidence {top['confidence']:.2f})")
        lines.append(f"- Heuristic agreement: {top['heuristic']['looks_like_deposit_cluster']} (score {top['heuristic']['heuristic_score']})")
        lines.append("- Top contributing features (importance-weighted proxy, NOT SHAP):")
        for f in top["top_features"]:
            lines.append(f"  - `{f['feature']}` = {f['value']} (weight {f['weight']})")

    lines += [
        "",
        f"## Grading (demo only -- would not appear in a real evidence pack)",
        f"- Ground truth VASP: {report['ground_truth_vasp_for_grading']}",
        f"- Top prediction correct: **{report['top_prediction_correct']}**",
        "",
        "*This output is a ranked probability with supporting evidence, not an assertion. Human investigator sign-off is required before any legal request is filed.*",
    ]

    out_path = OUTPUT_DIR / f"evidence_report_{'broken' if broken_trail else 'clean'}.md"
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broken-trail", action="store_true")
    args = parser.parse_args()

    result = run(broken_trail=args.broken_trail)
    print(json.dumps(result, indent=2))
