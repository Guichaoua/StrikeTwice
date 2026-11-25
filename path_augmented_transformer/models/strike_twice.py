"""Two-strike Edge--Vertex Transformer on graph and path occurrences."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from path_augmented_transformer.data_types import (
    EdgeVertexOutput,
    GraphPathBatch,
    PathEncoderOutput,
    gather_path_nodes,
    gather_path_pairs,
)
from path_augmented_transformer.models.edge_vertex_transformer import (
    EdgeVertexTransformer,
    PairInitialization,
    TriangularHeadConfig,
)
from path_augmented_transformer.models.evt_pool import encode_unique_graphs


def position_codes(
    max_path_length: int,
    include_absolute_position: bool,
    include_relative_position: bool,
    path_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Return coherent unary ``phi`` and binary ``psi`` position codes."""

    if path_length > max_path_length:
        raise ValueError(
            f"Path length {path_length} exceeds max_path_length="
            f"{max_path_length}."
        )
    if include_absolute_position:
        phi = torch.eye(max_path_length, device=device, dtype=dtype)[:path_length]
        source_phi = phi[:, None, :].expand(path_length, path_length, -1)
        target_phi = phi[None, :, :].expand(path_length, path_length, -1)
        components = [source_phi, target_phi]
    else:
        phi = torch.empty((path_length, 0), device=device, dtype=dtype)
        components = []
    if include_relative_position:
        positions = torch.arange(path_length, device=device, dtype=dtype)
        denominator = max(1, max_path_length - 1)
        relative = (positions[None, :] - positions[:, None]) / denominator
        components.append(relative.unsqueeze(-1))
    if components:
        psi = torch.cat(components, dim=-1)
    else:
        psi = torch.empty(
            (path_length, path_length, 0), device=device, dtype=dtype
        )
    return phi, psi


def select_occurrence_graph(
    graph: EdgeVertexOutput,
    batch: GraphPathBatch,
    max_path_length: int,
    include_absolute_position: bool,
    include_relative_position: bool,
    size_signals: Optional[tuple[Tensor, Tensor]] = None,
) -> EdgeVertexOutput:
    """Select graph states on ordered occurrences, retaining repeated visits.

    The appended equality bit is 1[p_s == p_t], not 1[s == t]. The latter
    is constructed separately by the path-stage incidence-aware MLP.
    Both bits are valid only where both path positions are unmasked.
    """

    path_length = batch.p.shape[1]
    phi, psi = position_codes(
        max_path_length,
        include_absolute_position,
        include_relative_position,
        path_length,
        device=graph.X.device,
        dtype=graph.X.dtype,
    )
    batch_size = batch.p.shape[0]
    phi = phi.unsqueeze(0).expand(batch_size, -1, -1)
    psi = psi.unsqueeze(0).expand(batch_size, -1, -1, -1)
    selected_X = gather_path_nodes(graph.X, batch.p, batch.path_mask)
    selected_Y = gather_path_pairs(graph.Y, batch.p, batch.path_mask)
    safe_p = batch.p.masked_fill(~batch.path_mask, 0)
    pair_mask = batch.path_mask.unsqueeze(2) & batch.path_mask.unsqueeze(1)
    same_graph_vertex = (
        safe_p.unsqueeze(2).eq(safe_p.unsqueeze(1)) & pair_mask
    ).to(dtype=graph.Y.dtype).unsqueeze(-1)
    selected_X = torch.cat((selected_X, phi), dim=-1)
    selected_Y = torch.cat((selected_Y, psi, same_graph_vertex), dim=-1)
    if size_signals is not None:
        graph_size, path_size = size_signals
        if graph_size.shape != (batch_size, 1) or path_size.shape != (batch_size, 1):
            raise ValueError("size signals must both have shape [B, 1].")
        unary_sizes = torch.cat((graph_size, path_size), dim=-1)
        unary_sizes = unary_sizes[:, None, :].expand(-1, path_length, -1)
        pair_sizes = torch.cat((graph_size, path_size), dim=-1)
        pair_sizes = pair_sizes[:, None, None, :].expand(
            -1, path_length, path_length, -1
        )
        selected_X = torch.cat((selected_X, unary_sizes), dim=-1)
        selected_Y = torch.cat((selected_Y, pair_sizes), dim=-1)
    return EdgeVertexOutput(
        X=selected_X.masked_fill(~batch.path_mask.unsqueeze(-1), 0),
        Y=selected_Y.masked_fill(~pair_mask.unsqueeze(-1), 0),
    )


class StrikeTwice(nn.Module):
    r"""Apply EVT globally, pull back both streams, then apply EVT again.

    The first EVT is independent of ``P`` and can be evaluated once for all
    queries sharing a graph.  The exact selector constructs an occurrence
    graph with

    ``X_hat[s] = [X_graph[p_s]; phi(s)]`` and
    ``Y_hat[s,t] = [Y_graph[p_s,p_t]; phi(s); phi(t); rho(t-s); 1[p_s=p_t]]``.

    The second EVT operates on the ``k`` path occurrences and all ``k^2``
    ordered occurrence pairs.  Its vertex states are exposed as ``Z`` and its
    pair states as ``R`` in the common :class:`PathEncoderOutput` interface.

    ``include_size_signals`` is an opt-in masked-padding control. It appends
    normalized graph size to the first strike and normalized graph/path sizes
    to both pulled-back streams. ``theory_mode`` independently replaces
    LayerNorm by identity and GELU by ReLU in both EVT towers; padding remains
    masked in that mode.

    ``graph_pair_attention`` and ``path_pair_attention`` independently select
    the relational primitive of the first and second strikes.  Their defaults
    are the frozen triangular architecture and the only setting covered by the
    exact two-strike kernel theorem.  Setting either to ``'dense'`` replaces
    bridge-aligned triangular attention by ordinary self-attention over pair
    tokens while holding fixed depth, attention count and incidence coupling.
    The two switches expose the complete TT/TD/DT/DD mechanism factorial.

    ``graph_triangular_head_config`` and ``path_triangular_head_config`` are
    independent because branching motifs in the original DAG and ordered
    compositions in the occurrence graph need not receive the same head
    allocation.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_h: int,
        max_path_length: int,
        graph_num_layers: int = 2,
        path_num_layers: int = 2,
        num_heads: int = 4,
        graph_feedforward_dim: Optional[int] = None,
        path_feedforward_dim: Optional[int] = None,
        node_projection: str = "linear",
        pair_projection: str = "linear",
        include_absolute_position: bool = True,
        include_relative_position: bool = True,
        include_size_signals: bool = False,
        graph_size_reference: Optional[int] = None,
        theory_mode: bool = False,
        graph_pair_attention: str = "triangular",
        path_pair_attention: str = "triangular",
        graph_triangular_head_config: TriangularHeadConfig = "chain_only",
        path_triangular_head_config: TriangularHeadConfig = "chain_only",
        pair_initialization: PairInitialization = "separate",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        triangular_residual_init: Optional[float] = None,
        graph_triangular_residual_init: Optional[float] = None,
        path_triangular_residual_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if max_path_length < 1:
            raise ValueError("max_path_length must be positive.")
        if graph_num_layers < 1 or path_num_layers < 1:
            raise ValueError("Both EVT strikes require at least one layer.")
        if graph_pair_attention not in {"triangular", "dense"}:
            raise ValueError(
                "graph_pair_attention must be 'triangular' or 'dense'."
            )
        if path_pair_attention not in {"triangular", "dense"}:
            raise ValueError(
                "path_pair_attention must be 'triangular' or 'dense'."
            )
        if triangular_residual_init is not None and (
            graph_triangular_residual_init is not None
            or path_triangular_residual_init is not None
        ):
            raise ValueError(
                "triangular_residual_init cannot be combined with a "
                "strike-specific triangular residual initialization."
            )
        if include_size_signals and (
            graph_size_reference is None or graph_size_reference < 1
        ):
            raise ValueError(
                "include_size_signals requires a positive graph_size_reference."
            )
        self.d_h = d_h
        self.max_path_length = max_path_length
        self.include_absolute_position = include_absolute_position
        self.include_relative_position = include_relative_position
        self.include_size_signals = include_size_signals
        self.graph_size_reference = graph_size_reference
        self.theory_mode = theory_mode
        self.graph_pair_attention = graph_pair_attention
        self.path_pair_attention = path_pair_attention
        self.graph_triangular_head_config = graph_triangular_head_config
        self.path_triangular_head_config = path_triangular_head_config
        self.triangular_residual_init = triangular_residual_init
        self.graph_triangular_residual_init = (
            triangular_residual_init
            if graph_triangular_residual_init is None
            else graph_triangular_residual_init
        )
        self.path_triangular_residual_init = (
            triangular_residual_init
            if path_triangular_residual_init is None
            else path_triangular_residual_init
        )
        graph_size_dim = 1 if include_size_signals else 0
        self.graph_backbone = EdgeVertexTransformer(
            d_x=d_x + graph_size_dim,
            d_y=d_y + graph_size_dim,
            d_h=d_h,
            num_layers=graph_num_layers,
            num_heads=num_heads,
            feedforward_dim=graph_feedforward_dim,
            node_projection=node_projection,
            pair_projection=pair_projection,
            dropout=dropout,
            attention_dropout=attention_dropout,
            theory_mode=theory_mode,
            pair_attention=graph_pair_attention,
            triangular_head_config=graph_triangular_head_config,
            pair_initialization=pair_initialization,
            triangular_residual_init=(
                self.graph_triangular_residual_init
                if graph_pair_attention == "triangular"
                else None
            ),
        )
        # Keep ``backbone`` as the conventional name used by diagnostics.
        self.backbone = self.graph_backbone
        pair_position_dim = 2 * max_path_length if include_absolute_position else 0
        if include_relative_position:
            pair_position_dim += 1
        selected_node_dim = d_h + (
            max_path_length if include_absolute_position else 0
        )
        selected_pair_dim = d_h + pair_position_dim + 1
        if include_size_signals:
            selected_node_dim += 2
            selected_pair_dim += 2
        self.path_backbone = EdgeVertexTransformer(
            d_x=selected_node_dim,
            d_y=selected_pair_dim,
            d_h=d_h,
            num_layers=path_num_layers,
            num_heads=num_heads,
            feedforward_dim=path_feedforward_dim,
            node_projection="linear",
            pair_projection="linear",
            dropout=dropout,
            attention_dropout=attention_dropout,
            theory_mode=theory_mode,
            pair_attention=path_pair_attention,
            triangular_head_config=path_triangular_head_config,
            pair_initialization=pair_initialization,
            triangular_residual_init=(
                self.path_triangular_residual_init
                if path_pair_attention == "triangular"
                else None
            ),
        )

    def _size_signals(self, batch: GraphPathBatch) -> tuple[Tensor, Tensor]:
        """Return normalized valid graph and path cardinalities as [B, 1]."""

        if not self.include_size_signals or self.graph_size_reference is None:
            raise RuntimeError("Size signals are disabled for this model.")
        graph_size = batch.node_mask.sum(dim=1, keepdim=True).to(batch.X.dtype)
        graph_size = graph_size / float(self.graph_size_reference)
        path_size = batch.path_mask.sum(dim=1, keepdim=True).to(batch.X.dtype)
        path_size = path_size / float(self.max_path_length)
        return graph_size, path_size

    def _graph_batch(self, batch: GraphPathBatch) -> GraphPathBatch:
        """Append the graph-size signal without making the graph path-dependent."""

        if not self.include_size_signals:
            return batch
        graph_size, _ = self._size_signals(batch)
        n = batch.X.shape[1]
        node_size = graph_size[:, None, :].expand(-1, n, -1)
        pair_size = graph_size[:, None, None, :].expand(-1, n, n, -1)
        return GraphPathBatch(
            X=torch.cat((batch.X, node_size), dim=-1),
            Y=torch.cat((batch.Y, pair_size), dim=-1),
            p=batch.p,
            node_mask=batch.node_mask,
            path_mask=batch.path_mask,
            graph_index=batch.graph_index,
        )

    def _position_codes(
        self,
        path_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return coherent unary ``phi`` and binary ``psi`` position codes."""

        return position_codes(
            self.max_path_length,
            self.include_absolute_position,
            self.include_relative_position,
            path_length,
            device=device,
            dtype=dtype,
        )

    def select_occurrence_graph(
        self,
        graph: EdgeVertexOutput,
        batch: GraphPathBatch,
    ) -> EdgeVertexOutput:
        """Pull graph vertex and pair memories back along ``P`` and ``P x P``."""

        size_signals = self._size_signals(batch) if self.include_size_signals else None
        return select_occurrence_graph(
            graph,
            batch,
            self.max_path_length,
            self.include_absolute_position,
            self.include_relative_position,
            size_signals=size_signals,
        )

    def encode(self, batch: GraphPathBatch) -> PathEncoderOutput:
        graph = encode_unique_graphs(self.graph_backbone, self._graph_batch(batch))
        selected = self.select_occurrence_graph(graph, batch)
        occurrence = self.path_backbone(
            selected.X,
            selected.Y,
            batch.path_mask,
        )
        return PathEncoderOutput(
            X=graph.X,
            Y=graph.Y,
            Z=occurrence.X,
            R=occurrence.Y,
        )

    def forward(self, batch: GraphPathBatch) -> PathEncoderOutput:
        """Encode the graphs and paths in a padded batch."""

        return self.encode(batch)
