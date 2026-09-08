"""
etherscan_client.py

Reference implementation of the live chain-ingestion layer described in the
ChainTrace pitch deck ("Live graph pull -- k-hop sub-graph pulled from public
Etherscan / Blockchair APIs").

NOTE FOR THE DEMO ENVIRONMENT:
This module is NOT invoked by pipeline/run_demo.py. The sandbox this prototype
was built in only allows egress to a fixed allow-list of domains (pypi, npm,
github, etc.) and api.etherscan.io is not on it, so live calls fail closed
here. The demo instead runs against ingestion/synthetic_data.py, which
reproduces the same graph shape (peel-chain hops, mixer fan-out, deposit-
cluster fan-in, VASP hot wallet) that a real pull would return.

To go live: set ETHERSCAN_API_KEY and swap the call in run_demo.py from
`generate_case()` to `fetch_khop_subgraph(address, k=2)`. Nothing downstream
(feature extraction, embeddings, classifier) needs to change -- both paths
produce the same edge-list schema, defined in graph/build_graph.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
DEFAULT_TIMEOUT_S = 10
RATE_LIMIT_SLEEP_S = 0.25  # free tier: ~5 req/sec


@dataclass
class Txn:
    from_addr: str
    to_addr: str
    value_wei: int
    timestamp: int
    tx_hash: str


class EtherscanClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ETHERSCAN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ETHERSCAN_API_KEY not set. Get a free key at "
                "https://etherscan.io/apis and export it before running "
                "against the live API."
            )
        self._session = requests.Session()

    def get_txns(self, address: str, start_block: int = 0, end_block: int = 99_999_999) -> list[Txn]:
        """Pull the full normal-transaction list for one address (one API call)."""
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": "asc",
            "apikey": self.api_key,
        }
        resp = self._session.get(ETHERSCAN_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "1" and body.get("message") != "No transactions found":
            raise RuntimeError(f"Etherscan error for {address}: {body.get('message')}")

        time.sleep(RATE_LIMIT_SLEEP_S)  # respect free-tier rate limit
        return [
            Txn(
                from_addr=t["from"].lower(),
                to_addr=t["to"].lower(),
                value_wei=int(t["value"]),
                timestamp=int(t["timeStamp"]),
                tx_hash=t["hash"],
            )
            for t in body.get("result", [])
            if t.get("to")  # skip contract-creation rows with no `to`
        ]

    def fetch_khop_subgraph(
        self,
        root_address: str,
        k: int = 2,
        max_neighbors_per_hop: int = 40,
    ) -> list[Txn]:
        """
        Breadth-first expansion out to k hops from root_address.

        `max_neighbors_per_hop` is the hub-pruning control mentioned in the
        deck's risk table ("Hub wallets blow up graph size" -> degree-capped
        BFS). Without this cap, one exchange hot wallet in the neighborhood
        can pull in tens of thousands of unrelated addresses.
        """
        seen_addrs = {root_address.lower()}
        frontier = [root_address.lower()]
        all_txns: list[Txn] = []

        for _hop in range(k):
            next_frontier: list[str] = []
            for addr in frontier:
                txns = self.get_txns(addr)[:max_neighbors_per_hop]
                all_txns.extend(txns)
                for t in txns:
                    other = t.to_addr if t.from_addr == addr else t.from_addr
                    if other not in seen_addrs:
                        seen_addrs.add(other)
                        next_frontier.append(other)
            frontier = next_frontier

        return all_txns
