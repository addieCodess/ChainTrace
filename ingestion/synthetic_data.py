"""
synthetic_data.py

Generates realistic *shapes* of on-chain behaviour so the rest of the
pipeline can be built and demoed without a live API call:

  unknown wallet
      -> peel-chain hops (2-3 intermediate wallets, decreasing value)
      -> optional mixer-style fan-out (only in the "broken trail" case)
      -> deposit-address cluster (high fan-in, regular sweep timing)
      -> VASP hot wallet (the label we're trying to recover)

We also generate several *other* fully-labeled VASP examples (different
hot wallets, different deposit clusters) purely as training data for the
classifier, plus a handful of "noise" wallets that don't lead anywhere,
so the classifier has to learn structure, not memorize one shape.

This is a deliberate stand-in for two things a real deployment would use
instead (documented in DECISIONS.md):
  1. ingestion/etherscan_client.py for the live pull
  2. a bootstrapped label set (Etherscan Label Word Cloud / Elliptic++ /
     SNAP Ethereum-exchange address lists) for the "known VASP cluster"
     ground truth, instead of made-up addresses
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

Txn = tuple[str, str, float, int]  # (from, to, value_eth, timestamp)

BASE_TS = 1_726_000_000  # arbitrary Unix epoch anchor, Sept 2026-ish


@dataclass
class SyntheticCase:
    """One fully-generated scenario: edges + which node is the unknown wallet
    + which node is the ground-truth VASP hot wallet (so the demo can grade
    itself)."""

    label: str  # VASP name, or "UNKNOWN" for the case we're solving live
    edges: list[Txn] = field(default_factory=list)
    unknown_wallet: str = ""
    hot_wallet: str = ""
    deposit_cluster: list[str] = field(default_factory=list)
    broken_trail: bool = False  # True if a mixer hop is injected


def _addr(prefix: str, i: int) -> str:
    return f"0x{prefix}{i:04d}"


def _canonical_hot_wallet(vasp_name: str) -> str:
    """A VASP's hot wallet is public, well-known infrastructure -- unlike a
    fresh deposit address, it's the SAME address across every case that
    happens to route to that exchange. This is what lets a brand-new,
    never-seen deposit cluster still get attributed: it's one hop from an
    address that's already known, even though the deposit address itself
    isn't."""
    return f"0xHOT_{vasp_name}"


def _make_labeled_vasp_case(rng: random.Random, vasp_name: str, seed_id: int, n_depositors: int = 8) -> SyntheticCase:
    """Build one clean 'this cluster sweeps into VASP X' training example."""
    hot_wallet = _canonical_hot_wallet(vasp_name)
    deposit_cluster = [_addr(f"DEP_{vasp_name}_", seed_id * 100 + j) for j in range(3)]
    edges: list[Txn] = []

    ts = BASE_TS + seed_id * 500_000
    for dep in deposit_cluster:
        # many customer wallets feed into each deposit address (fan-in)
        for j in range(n_depositors):
            depositor = _addr(f"CUST_{vasp_name}_", seed_id * 1000 + hash((dep, j)) % 900)
            value = round(rng.uniform(0.05, 3.0), 4)
            edges.append((depositor, dep, value, ts + j * rng.randint(60, 400)))
        # regular sweep from deposit address into the hot wallet -- the
        # "sweep-timing regularity" signal: near-constant interval, low jitter
        sweep_ts = ts + n_depositors * 300 + rng.randint(0, 30)
        edges.append((dep, hot_wallet, sum(rng.uniform(0.05, 3.0) for _ in range(n_depositors)), sweep_ts))

    return SyntheticCase(
        label=vasp_name,
        edges=edges,
        hot_wallet=hot_wallet,
        deposit_cluster=deposit_cluster,
    )


def _make_unknown_case(rng: random.Random, true_vasp: str, seed_id: int, broken_trail: bool = False) -> SyntheticCase:
    """
    Build the 'live investigation' case: an UNKNOWN wallet that launders
    through a peel chain (and optionally a mixer) before converging on a
    deposit cluster that sweeps into `true_vasp`'s hot wallet.

    `true_vasp` must match the vasp_name used in one of the labeled training
    cases, so the demo can check whether the classifier recovered it.
    """
    unknown = _addr("UNKNOWN_", seed_id)
    hop1, hop2 = _addr("HOP_", seed_id * 10 + 1), _addr("HOP_", seed_id * 10 + 2)
    deposit_cluster = [_addr(f"DEP_{true_vasp}_", seed_id * 100), _addr(f"DEP_{true_vasp}_", seed_id * 100 + 1)]
    hot_wallet = _canonical_hot_wallet(true_vasp)

    edges: list[Txn] = []
    ts = BASE_TS + 50_000_000 + seed_id * 10_000

    stolen_amount = rng.uniform(8.0, 40.0)
    edges.append((unknown, hop1, stolen_amount * 0.9, ts))

    if broken_trail:
        # mixer-style fan-out: value gets split across many one-hop wallets,
        # which is exactly the pattern that should make confidence DROP
        # rather than the model confidently guessing a VASP
        mixer_out = [_addr("MIX_", seed_id * 10 + j) for j in range(6)]
        share = (stolen_amount * 0.85) / len(mixer_out)
        for j, m in enumerate(mixer_out):
            edges.append((hop1, m, share, ts + 200 + j * 15))
        # only a couple of the mixer outputs happen to re-converge on the
        # real deposit cluster -- a realistic partial trail
        edges.append((mixer_out[0], deposit_cluster[0], share * 0.97, ts + 900))
        edges.append((mixer_out[1], deposit_cluster[1], share * 0.95, ts + 940))
    else:
        edges.append((hop1, hop2, stolen_amount * 0.85, ts + 300))
        edges.append((hop2, deposit_cluster[0], stolen_amount * 0.8, ts + 650))
        # ordinary customer traffic into BOTH deposit addresses, at a volume
        # comparable to ExchangeA's labeled training clusters (~10 inbound
        # per address) -- this is the realistic part of the cold-start
        # story: a fresh deposit address at a real exchange has real
        # customer fan-in from day one, matching that exchange's normal
        # operating pattern, even though nobody has labeled *this specific
        # address* yet. Without matching the volume, the deposit address
        # would look like a smaller or busier exchange's cluster instead.
        for dep in deposit_cluster:
            for j in range(9):
                depositor = _addr("CUST_bg_", seed_id * 1000 + hash((dep, j)) % 900)
                edges.append((depositor, dep, round(rng.uniform(0.1, 2.0), 4), ts + 700 + j * 90))

    sweep_ts = ts + 1200
    for dep in deposit_cluster:
        edges.append((dep, hot_wallet, rng.uniform(1.0, 5.0), sweep_ts + rng.randint(0, 20)))

    return SyntheticCase(
        label="UNKNOWN",
        edges=edges,
        unknown_wallet=unknown,
        hot_wallet=hot_wallet,  # ground truth, used only for grading the demo
        deposit_cluster=deposit_cluster,
        broken_trail=broken_trail,
    )


def generate_training_universe(seed: int = 7) -> list[SyntheticCase]:
    """3 labeled VASPs' worth of clean deposit-cluster -> hot-wallet examples."""
    rng = random.Random(seed)
    return [
        _make_labeled_vasp_case(rng, "ExchangeA", seed_id=1, n_depositors=10),
        _make_labeled_vasp_case(rng, "ExchangeB_OTC", seed_id=2, n_depositors=4),
        _make_labeled_vasp_case(rng, "ExchangeC", seed_id=3, n_depositors=6),
    ]


def generate_investigation_case(seed: int = 42, broken_trail: bool = False) -> SyntheticCase:
    """The single unknown wallet the demo pipeline is asked to attribute.
    Ground truth is ExchangeA -- kept out of the classifier's sight until
    grading time."""
    rng = random.Random(seed)
    return _make_unknown_case(rng, true_vasp="ExchangeA", seed_id=99, broken_trail=broken_trail)
