"""Profile tiled EVT graph encoding and path queries on a shared cache."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from path_augmented_transformer import (
    GraphPathBatch,
    PathEncoderOutput,
    ScalarPathHead,
    StrikeTwice,
)
from path_augmented_transformer.atlas_profile import (
    build_atlas_profile,
    make_synthetic_atlas_request,
)


WIDTH = 4
MIN_POSITION_CAPACITY = 32
FEATURE_TILE = 1
SHORT_QUERY_TILE = 16
LONG_QUERY_TILE = 1024
MIB = 1024**2
PAPER_MODEL_SEED = 20261112
PAPER_GRAPH_HEADS = 4


def toy_graph(num_nodes):
    edges = [[i, (i + step) % num_nodes] for i in range(num_nodes) for step in (1, 3)]
    paths = [
        [(start + step * i) % num_nodes for i in range(8)]
        for start, step in [(0, 1), (3, 3)]
    ]
    return dict(name="Synthetic directed ring", num_nodes=num_nodes, edges=edges, paths=paths)


def load_graph(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    n = graph["num_nodes"]
    if type(n) is not int or n < 2:
        raise ValueError(f"{path}: num_nodes must be an integer of at least two")
    edges = set(map(tuple, graph["edges"]))
    paths = graph["paths"]
    if not edges or any(len(edge) != 2 for edge in edges):
        raise ValueError(f"{path}: expected nonempty directed edges [source, target]")
    if any(type(i) is not int or not 0 <= i < n for edge in edges for i in edge):
        raise ValueError(f"{path}: edge vertex outside [0, {n})")
    if not paths or any(len(query) < 2 for query in paths):
        raise ValueError(f"{path}: expected paths of at least two occurrences")
    if any(type(i) is not int for query in paths for i in query):
        raise ValueError(f"{path}: path vertices must be integers")
    if any((u, v) not in edges for query in paths for u, v in zip(query, query[1:])):
        raise ValueError(f"{path}: each consecutive path pair must be a directed edge")
    return graph


def make_request(graph, device):
    n = graph["num_nodes"]
    features = torch.ones((1, n, 1), device=device)
    adjacency = torch.zeros((1, n, n, 1), device=device)
    sources, targets = torch.tensor(graph["edges"], device=device).T
    adjacency[0, sources, targets, 0] = 1
    node_mask = torch.ones((1, n), dtype=torch.bool, device=device)
    queries = []
    for path in graph["paths"]:
        indices = torch.tensor([path], dtype=torch.long, device=device)
        queries.append(GraphPathBatch(features, adjacency, indices, node_mask, indices >= 0))

    torch.manual_seed(7)
    model = StrikeTwice(
        d_x=1,
        d_y=1,
        d_h=WIDTH,
        max_path_length=max(MIN_POSITION_CAPACITY, max(len(path) for path in graph["paths"])),
        graph_num_layers=1,
        path_num_layers=1,
        num_heads=1,
    ).to(device).eval()
    return model, ScalarPathHead(d_h=WIDTH).to(device).eval(), queries


def enable_tiles(model, source_tile):
    model.graph_backbone.enable_tiled_attention(
        source_tile=source_tile,
        feature_tile=FEATURE_TILE,
        short_query_tile=SHORT_QUERY_TILE,
        long_query_tile=LONG_QUERY_TILE,
    )


def encode(model, query):
    return model.graph_backbone(query.X, query.Y, query.node_mask)


def predict(model, head, cache, query):
    selected = model.select_occurrence_graph(cache, query)
    occurrence = model.path_backbone(selected.X, selected.Y, query.path_mask)
    states = PathEncoderOutput(cache.X, cache.Y, occurrence.X, occurrence.Y)
    return head(states, query.path_mask).prediction


def compare_small_graph(device, source_tile):
    model, head, queries = make_request(toy_graph(17), device)
    reference = [head(model(query), query.path_mask).prediction for query in queries]
    dense_cache = encode(model, queries[0])
    enable_tiles(model, source_tile)
    cache = encode(model, queries[0])
    torch.testing.assert_close(cache.X, dense_cache.X, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(cache.Y, dense_cache.Y, atol=1e-6, rtol=2e-5)
    for query, expected in zip(queries, reference):
        torch.testing.assert_close(
            predict(model, head, cache, query), expected, atol=1e-6, rtol=2e-5
        )
    print("Small graph: dense and tiled caches agree, and cached predictions match recomputation.")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def profile(device, build_cache, read_paths, *, n, num_paths, heads, args):
    encoding_times, decoding_times, cuda_peaks = [], [], []
    for repetition in range(args.warmups + args.repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = perf_counter()
        cache = build_cache()
        synchronize(device)
        encoded = perf_counter()
        predictions = read_paths(cache)
        synchronize(device)
        decoded = perf_counter()
        if device.type == "cuda" and repetition >= args.warmups:
            cuda_peaks.append(torch.cuda.max_memory_allocated(device))
        if not torch.isfinite(predictions).all():
            raise RuntimeError("Non-finite prediction")
        if repetition >= args.warmups:
            encoding_times.append(encoded - start)
            decoding_times.append(decoded - encoded)
        cache_bytes = sum(state.numel() * state.element_size() for state in (cache.X, cache.Y))
        item_bytes = cache.Y.element_size()
        del cache, predictions

    plain_scores = heads * n**3 * item_bytes
    tiled_scores = heads * min(args.source_tile, n) * n**2 * item_bytes
    print(f"  Build cache once: {mean(encoding_times):.3f} s")
    print(f"  Read {num_paths} cached paths: {mean(decoding_times) * 1000:.2f} ms total")
    print(f"  Mean complete request time: {mean(encoding_times) + mean(decoding_times):.3f} s")
    print(f"  Logical vertex + pair cache: {cache_bytes / MIB:.2f} MiB")
    print(f"  One triangular score tensor: {plain_scores / MIB:.2f} MiB untiled, "
          f"{tiled_scores / MIB:.2f} MiB per tile (shape calculation)")
    if cuda_peaks:
        print(f"  Peak allocated CUDA memory: {max(cuda_peaks) / MIB:.2f} MiB")


def run_graph(graph, device, args):
    n = graph["num_nodes"]
    print(f"\n{graph['name']}: {n:,} vertices, {len(graph['edges']):,} directed edges", flush=True)
    model, head, queries = make_request(graph, device)
    enable_tiles(model, args.source_tile)
    profile(
        device,
        lambda: encode(model, queries[0]),
        lambda cache: torch.stack([predict(model, head, cache, query) for query in queries]),
        n=n, num_paths=len(queries), heads=1, args=args,
    )


def atlas_topology_request(graph):
    """Put an observed topology and routes into the paper's input shapes."""
    n = graph["num_nodes"]
    vertices = torch.zeros(1, n, 7)
    pairs = torch.zeros(1, n, n, 6)
    sources, targets = torch.tensor(graph["edges"]).T
    pairs[0, sources, targets, 0] = 1
    queries = graph["paths"]
    p = torch.full((len(queries), max(map(len, queries))), -1, dtype=torch.long)
    for row, query in enumerate(queries):
        p[row, :len(query)] = torch.tensor(query)
    return dict(X=vertices, Y=pairs, p=p, node_mask=torch.ones(1, n, dtype=torch.bool),
                path_mask=p >= 0, prior=torch.zeros_like(p, dtype=torch.float32))


def enable_paper_tiles(model, source_tile):
    model.encoder.graph_backbone.enable_tiled_attention(
        source_tile=source_tile, feature_tile=2, short_query_tile=32, long_query_tile=1024
    )


def compare_paper_outputs(device, source_tile):
    torch.manual_seed(PAPER_MODEL_SEED)
    model = build_atlas_profile().to(device).eval()
    batch = make_synthetic_atlas_request(8)
    X, Y, mask, p, path_mask = (
        batch[key].to(device) for key in ("X", "Y", "node_mask", "p", "path_mask")
    )
    expected = model(X, Y, mask, p, path_mask)
    enable_paper_tiles(model, source_tile)
    cache = model.encode_graph(X, Y, mask)
    actual = model.decode_paths(cache, p, path_mask)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    print("Small paper-shaped graph: native and tiled occurrence outputs agree.")


def run_paper_profile(graph, device, args):
    if graph is None:
        batch = make_synthetic_atlas_request(args.nodes)
        name = "Paper synthetic graph"
    else:
        batch = atlas_topology_request(graph)
        name = graph["name"] + " (observed topology, placeholder features)"
    X, Y, mask, p, path_mask = (
        batch[key].to(device) for key in ("X", "Y", "node_mask", "p", "path_mask")
    )
    n = X.shape[1]
    torch.manual_seed(PAPER_MODEL_SEED)
    model = build_atlas_profile().to(device).eval()
    enable_paper_tiles(model, args.source_tile)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n{name}: {n:,} vertices, {params:,} parameters, "
          f"one physical batch of {p.shape[0]} paths", flush=True)
    profile(
        device,
        lambda: model.encode_graph(X, Y, mask),
        lambda cache: model.decode_paths(cache, p, path_mask),
        n=n, num_paths=p.shape[0], heads=PAPER_GRAPH_HEADS, args=args,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", type=Path, help="JSON routing graph fixtures")
    parser.add_argument("--paper", action="store_true", help="use the full Atlas model")
    parser.add_argument("--nodes", type=int, help="synthetic graph size (default: 64)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--source-tile", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if min(args.source_tile, args.threads, args.repeats) < 1 or args.warmups < 0:
        parser.error("Tile size, threads and repeats must be positive and warmups nonnegative")
    if args.nodes is not None and (args.nodes < 8 or args.graphs):
        parser.error("--nodes requires a synthetic graph and at least eight vertices")
    args.nodes = args.nodes or 64
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")

    width = 64 if args.paper else WIDTH
    print(f"Device: {device}, FP32, width {width}, source tile {args.source_tile}, "
          f"{args.warmups} warmups, {args.repeats} timed runs")
    graphs = [load_graph(path) for path in args.graphs]
    if args.paper:
        compare_paper_outputs(device, args.source_tile)
        for graph in graphs or [None]:
            run_paper_profile(graph, device, args)
    else:
        compare_small_graph(device, args.source_tile)
        for graph in graphs or [toy_graph(args.nodes)]:
            run_graph(graph, device, args)


if __name__ == "__main__":
    main()
