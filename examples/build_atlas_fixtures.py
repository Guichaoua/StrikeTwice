"""Build topology-only examples from the pinned RIPE Atlas Region Meshes snapshots."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import heapq
import ipaddress
import json
from pathlib import Path


DATE = "2026-08-01"
SIZE = 1024
SOURCE = "https://ftp.ripe.net/ripe/atlas/region-meshes"
TERMS = "https://www.ripe.net/about-us/legal/ripe-atlas-service-terms-and-conditions/"
SNAPSHOTS = {
    "DE": "0c18d9ed43193c96c727f09be7346bf4bb1a4f0d09a3353de6c79922ab2b8291",
    "US": "61d98a5bdc65c00e432eff2aeed24baa430286b0860212f1ba8e426612ef76dd",
}
NONPUBLIC = tuple(map(ipaddress.IPv4Network, (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
    "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4",
)))


@lru_cache(maxsize=32768)
def is_public(address):
    try:
        ip = ipaddress.IPv4Address(address)
    except (ValueError, TypeError):
        return False
    return not any(ip in block for block in NONPUBLIC)


def consecutive_runs(values):
    run = []
    for value in values:
        if value is None:
            if len(run) >= 2:
                yield tuple(run)
            run = []
        else:
            run.append(value)
    if len(run) >= 2:
        yield tuple(run)


def observed_hops(trace):
    previous_hop = None
    for hop in trace.get("result", []):
        ttl = hop.get("hop")
        if not isinstance(ttl, int) or (previous_hop is not None and ttl != previous_hop + 1):
            yield None
        previous_hop = ttl if isinstance(ttl, int) else None
        if previous_hop is None:
            continue
        responders = Counter(
            address for reply in hop.get("result", [])
            if isinstance((address := reply.get("from")), str) and is_public(address)
        )
        yield min(responders, key=lambda node: (-responders[node], node)) if responders else None


def retained_vertices(edges, frequency):
    neighbors = defaultdict(set)
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    visited = set()
    for seed in sorted(neighbors, key=lambda node: (-frequency[node], node)):
        if seed in visited:
            continue
        selected = []
        frontier = [(-frequency[seed], seed)]
        visited.add(seed)
        while frontier and len(selected) < SIZE:
            _, node = heapq.heappop(frontier)
            selected.append(node)
            for candidate in neighbors[node] - visited:
                heapq.heappush(frontier, (-frequency[candidate], candidate))
            visited.update(neighbors[node])
        if len(selected) == SIZE:
            return selected
    raise ValueError(f"No observed component contains {SIZE} vertices")


def build_fixture(raw_root, region):
    basename = f"result_data_{region}_{DATE}_ipv4.jsonl.gz"
    source_file = raw_root / DATE / region / basename
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != SNAPSHOTS[region]:
        raise ValueError(f"Unexpected RIPE Atlas source digest for {region}")

    segments = []
    frequency = Counter()
    with gzip.open(source_file, mode="rt", encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            if trace.get("type") != "traceroute":
                continue
            runs = list(consecutive_runs(observed_hops(trace)))
            segments.extend(runs)
            frequency.update({node for run in runs for node in run})

    edges = {(u, v) for run in segments for u, v in zip(run, run[1:])}
    order = retained_vertices(edges, frequency)
    indices = {node: i for i, node in enumerate(order)}
    fixture_edges = sorted((indices[u], indices[v]) for u, v in edges if u in indices and v in indices)
    candidates = {
        path[:32] for segment in segments
        for path in consecutive_runs(indices.get(node) for node in segment)
        if len(path) >= 8
    }
    paths = sorted(candidates, key=lambda path: (-len(path), path))[:4]
    if len(paths) < 2:
        raise ValueError(f"Only {len(paths)} usable paths of at least eight occurrences for {region}")

    return {
        "name": f"atlas-{region.lower()}-1024",
        "num_nodes": SIZE,
        "edges": fixture_edges,
        "paths": paths,
        "provenance": {
            "publisher": "RIPE NCC",
            "collection": "RIPE Atlas Region Meshes",
            "region": region,
            "snapshot": DATE,
            "source": f"{SOURCE}/{DATE}/{region}.mesh/{basename}",
            "source_sha256": SNAPSHOTS[region],
            "terms": TERMS,
            "construction": (
                "Aggregate the regional snapshot across measurement destinations. "
                "Choose the most frequent public IPv4 responder per hop with lexical ties. "
                "Break paths at missing responders and TTL gaps, then connect consecutive hops. "
                "Grow a weakly connected set of 1024 observed vertices by descending trace "
                "frequency with lexical ties. Use consecutive observed subpaths within this set. "
                "Vertex IDs are local integers. No raw addresses or latency labels are included."
            ),
            "scope": "Topology-only demonstration from a regional aggregate outside the paper's per-destination benchmark.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Directory containing the downloaded DATE/REGION traceroute archives")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for region in SNAPSHOTS:
        fixture = build_fixture(args.raw_root, region)
        path = args.output_dir / (fixture["name"] + ".json")
        path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"{path.name}: {fixture['num_nodes']} vertices, {len(fixture['edges'])} edges, "
              f"query lengths {[len(p) for p in fixture['paths']]}")


if __name__ == "__main__":
    main()
