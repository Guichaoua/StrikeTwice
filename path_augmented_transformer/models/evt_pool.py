"""EVT-Pool: graph-contextualized vertices followed by path gathering."""

from typing import Optional

import torch
from torch import nn

from path_augmented_transformer.data_types import (
    EdgeVertexOutput,
    GraphPathBatch,
    PathEncoderOutput,
    gather_path_nodes,
)
from path_augmented_transformer.models.common import make_projection
from path_augmented_transformer.models.edge_vertex_transformer import (
    EdgeVertexTransformer,
    PairInitialization,
    TriangularHeadConfig,
)


def encode_unique_graphs(
    backbone: EdgeVertexTransformer,
    batch: GraphPathBatch,
) -> EdgeVertexOutput:
    """Encode each immutable graph once and broadcast it to its query rows."""

    if batch.graph_index is None:
        return backbone(batch.X, batch.Y, batch.node_mask)

    # MPS does not implement int64 scatter_reduce. Group only the tiny index
    # vector on CPU; all feature tensors and differentiable work stay on MPS.
    grouping_index = (
        batch.graph_index.cpu() if batch.X.device.type == "mps" else batch.graph_index
    )
    _, inverse = torch.unique(
        grouping_index,
        sorted=True,
        return_inverse=True,
    )
    positions = torch.arange(
        len(inverse), device=inverse.device, dtype=torch.long
    )
    first = torch.full(
        (int(inverse.max()) + 1,),
        len(inverse),
        device=inverse.device,
        dtype=torch.long,
    )
    first.scatter_reduce_(
        0,
        inverse,
        positions,
        reduce="amin",
        include_self=True,
    )
    first = first.to(batch.X.device)
    inverse = inverse.to(batch.X.device)
    unique_graph = backbone(
        batch.X[first],
        batch.Y[first],
        batch.node_mask[first],
    )
    return EdgeVertexOutput(
        X=unique_graph.X[inverse],
        Y=unique_graph.Y[inverse],
    )


class EVTPool(nn.Module):
    r"""Encode ``(X,Y)`` once, then gather :math:`Z_t=\phi_Z(X'_{p_t})`.

    The encoder itself does not pool or predict. The shared scalar head applies
    the exact masked mean to ``Z``; structured tasks may attach another head to
    the same typed ``(X', Y', Z)`` output.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_h: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: Optional[int] = None,
        node_projection: str = "linear",
        pair_projection: str = "linear",
        path_projection: str = "linear",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        triangular_head_config: TriangularHeadConfig = "chain_only",
        pair_initialization: PairInitialization = "separate",
    ) -> None:
        super().__init__()
        self.backbone = EdgeVertexTransformer(
            d_x=d_x,
            d_y=d_y,
            d_h=d_h,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            node_projection=node_projection,
            pair_projection=pair_projection,
            dropout=dropout,
            attention_dropout=attention_dropout,
            triangular_head_config=triangular_head_config,
            pair_initialization=pair_initialization,
        )
        self.path_projection = make_projection(d_h, d_h, path_projection)

    def encode(self, batch: GraphPathBatch) -> PathEncoderOutput:
        graph = encode_unique_graphs(self.backbone, batch)
        gathered = gather_path_nodes(graph.X, batch.p, batch.path_mask)
        Z = self.path_projection(gathered)
        Z = Z.masked_fill(~batch.path_mask.unsqueeze(-1), 0)
        return PathEncoderOutput(X=graph.X, Y=graph.Y, Z=Z)

    def forward(self, batch: GraphPathBatch) -> PathEncoderOutput:
        return self.encode(batch)
