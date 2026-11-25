"""StrikeTwice configuration and synthetic workload for the Atlas scale profile."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from path_augmented_transformer.data_types import EdgeVertexOutput, GraphPathBatch
from path_augmented_transformer.models.edge_vertex_transformer import EdgeVertexTransformer
from path_augmented_transformer.models.strike_twice import StrikeTwice


ATLAS_MODEL_SEED = 20261112
ATLAS_INPUT_SEED = 20261113
ATLAS_PARAMETER_COUNT = 778_897
PATH_CAPACITY = 64


@dataclass
class AtlasGraphCache(EdgeVertexOutput):
    """Encoded states and the validity mask of one shared graph."""

    node_mask: Tensor


class AtlasProfile(nn.Module):
    """Full-size two-EVT model with the original Atlas occurrence readout."""

    def __init__(self) -> None:
        super().__init__()
        width = 64
        self.encoder = StrikeTwice(
            d_x=7,
            d_y=6,
            d_h=width,
            max_path_length=PATH_CAPACITY,
            graph_num_layers=2,
            path_num_layers=2,
            num_heads=4,
            graph_feedforward_dim=128,
            path_feedforward_dim=128,
            graph_triangular_head_config="balanced",
            path_triangular_head_config="balanced",
        )
        # Replace the path stack as in the original Atlas model, which uses
        # eight path heads and four graph heads.
        self.encoder.path_backbone = EdgeVertexTransformer(
            d_x=width + PATH_CAPACITY,
            d_y=width + 2 * PATH_CAPACITY + 2,
            d_h=width,
            num_layers=2,
            num_heads=8,
            feedforward_dim=128,
            triangular_head_config="balanced",
        )
        self.head = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 1))

    def encode_graph(self, X: Tensor, Y: Tensor, node_mask: Tensor) -> AtlasGraphCache:
        """Encode one fixed graph for subsequent batches of query paths."""

        if X.ndim != 3 or X.shape[0] != 1:
            raise ValueError("Atlas profiling expects one shared graph per request.")
        states = self.encoder.graph_backbone(X, Y, node_mask)
        return AtlasGraphCache(states.X, states.Y, node_mask)

    def decode_paths(self, cache: AtlasGraphCache, p: Tensor, path_mask: Tensor) -> Tensor:
        """Return the dimensionless corrections for the valid path positions."""

        query_count = p.shape[0]
        # Each query refers to the same graph. Expand creates views of the
        # cache without allocating one full pair tensor for each query.
        batch = GraphPathBatch(
            X=cache.X.expand(query_count, -1, -1),
            Y=cache.Y.expand(query_count, -1, -1, -1),
            p=p,
            node_mask=cache.node_mask.expand(query_count, -1),
            path_mask=path_mask,
        )
        selected = self.encoder.select_occurrence_graph(EdgeVertexOutput(batch.X, batch.Y), batch)
        nodes = self.encoder.path_backbone(selected.X, selected.Y, path_mask).X
        position = torch.arange(nodes.shape[1], device=nodes.device, dtype=nodes.dtype)
        position = position[None, :, None].expand(query_count, -1, -1) / PATH_CAPACITY
        return self.head(torch.cat((nodes, position), dim=-1)).squeeze(-1)

    def predict_rtt(
        self, cache: AtlasGraphCache, p: Tensor, path_mask: Tensor, prior: Tensor
    ) -> Tensor:
        """Apply the original conversion from dimensionless correction to RTT in ms."""

        if prior.shape != p.shape:
            raise ValueError("The prior must have one value per path occurrence.")
        return prior + 10.0 * self.decode_paths(cache, p, path_mask)

    def forward(self, X: Tensor, Y: Tensor, node_mask: Tensor, p: Tensor, path_mask: Tensor) -> Tensor:
        return self.decode_paths(self.encode_graph(X, Y, node_mask), p, path_mask)


def build_atlas_profile() -> AtlasProfile:
    """Build the 778,897-parameter scale-profile model with initialized weights."""

    return AtlasProfile()


def make_synthetic_atlas_request(
    n: int, *, k: int | None = None, queries: int | None = None, seed: int = ATLAS_INPUT_SEED
) -> dict[str, Tensor]:
    """Generate the synthetic profiling graph and its directed ring paths."""

    if type(n) is not int or n < 1:
        raise ValueError("The graph must have a positive integer number of vertices.")
    k = min(n, 16) if k is None else k
    queries = min(n, 16) if queries is None else queries
    if type(k) is not int or not 1 <= k <= min(n, PATH_CAPACITY):
        raise ValueError("Path length must be between 1 and the smaller of graph size and 64.")
    if type(queries) is not int or not 1 <= queries <= n:
        raise ValueError("The number of queries must be between 1 and the graph size.")

    generator = torch.Generator().manual_seed(seed + n)
    X = torch.randn((1, n, 7), generator=generator, dtype=torch.float32)
    Y = torch.randn((1, n, n, 6), generator=generator, dtype=torch.float32)
    adjacency = torch.rand((n, n), generator=generator, dtype=torch.float32) < 0.05
    adjacency.fill_diagonal_(False)
    ring = torch.randperm(n, generator=generator)
    adjacency[ring, ring.roll(-1)] = True
    starts = torch.randperm(n, generator=generator)[:queries]
    p = ring[(starts[:, None] + torch.arange(k)) % n]
    Y[..., 0] = adjacency

    return {
        "X": X,
        "Y": Y,
        "node_mask": torch.ones((1, n), dtype=torch.bool),
        "p": p,
        "path_mask": torch.ones_like(p, dtype=torch.bool),
        "prior": torch.zeros_like(p, dtype=torch.float32),
    }
