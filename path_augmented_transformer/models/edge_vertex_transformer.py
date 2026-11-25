"""Readable dense PyTorch reference for the Edge--Vertex Transformer.

The implementation follows the frozen two-stage EVT layer from the paper:

1. parallel intra-stream updates ``X <- X`` and triangular ``Y <- Y``;
2. parallel dense cross-stream updates ``X <- Y`` and ``Y <- X``.

Each stage first normalizes each participating stream once.  All Q/K/V
projections with the same source reuse that normalized state.  After each
attention phase, an incidence-aware pointwise block updates vertices from
their diagonal pairs and ordered pairs from their two endpoints.  Its skip is
the ordinary identity residual; the local context only enters through the
pre-LayerNorm MLP.  This module intentionally materializes the cubic
triangular logits and is a correctness reference, not the future
Triton/FlashAttention implementation.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from path_augmented_transformer.data_types import EdgeVertexOutput
from path_augmented_transformer.models.common import attention_softmax, make_projection


# Manuscript convention: right arrow = stored Y[i,j]; left arrow = Y[j,i].
# These are views of ordered pairs, not directions of an assignment/update.
PairView = Literal["stored", "transpose"]
PairHeadDirection = Tuple[PairView, PairView]
PairInitialization = Literal["separate", "endpoint_lift"]
TriangularHeadConfig = Literal[
    "chain_only",
    "balanced",
    "dag_oriented",
]

TRIANGULAR_HEAD_CONFIGS: Tuple[TriangularHeadConfig, ...] = (
    "chain_only",
    "balanced",
    "dag_oriented",
)


def make_triangular_head_directions(
    num_heads: int,
    configuration: TriangularHeadConfig = "chain_only",
) -> Tuple[PairHeadDirection, ...]:
    """Return one explicit ordered-pair orientation per attention head.

    ``chain_only`` preserves the historical Edge Transformer composition in
    every head.  The two directional controls require a head count divisible
    by four so their allocations are exact rather than silently rounded:

    - ``balanced`` assigns equal width to all four orientations;
    - ``dag_oriented`` assigns half the width to directed chains and one
      quarter each to common-target and common-source wedges.

    In manuscript order (right/right, right/left, left/right, left/left),
    the four contractions are A@A, A@A.T, A.T@A, and A.T@A.T.
    """

    if num_heads < 1:
        raise ValueError("num_heads must be positive.")
    if configuration not in TRIANGULAR_HEAD_CONFIGS:
        raise ValueError(
            "triangular_head_config must be one of "
            f"{TRIANGULAR_HEAD_CONFIGS}."
        )
    stored: PairView = "stored"
    transpose: PairView = "transpose"
    chain = (stored, stored)
    if configuration == "chain_only":
        return (chain,) * num_heads
    if num_heads % 4 != 0:
        raise ValueError(
            f"{configuration!r} requires num_heads to be divisible by 4."
        )
    quarter = num_heads // 4
    common_target = (stored, transpose)
    common_source = (transpose, stored)
    reverse_chain = (transpose, transpose)
    if configuration == "balanced":
        return (
            (chain,) * quarter
            + (common_target,) * quarter
            + (common_source,) * quarter
            + (reverse_chain,) * quarter
        )
    return (
        (chain,) * (2 * quarter)
        + (common_target,) * quarter
        + (common_source,) * quarter
    )


def _validate_head_directions(
    head_directions: Sequence[PairHeadDirection],
    num_heads: int,
) -> Tuple[PairHeadDirection, ...]:
    """Validate and freeze a per-head directional assignment."""

    if len(head_directions) != num_heads:
        raise ValueError("head_directions must contain exactly num_heads entries.")
    valid_views = {"stored", "transpose"}
    frozen = []
    for direction in head_directions:
        if len(direction) != 2 or any(view not in valid_views for view in direction):
            raise ValueError(
                "Each head direction must be a pair of 'stored' or 'transpose'."
            )
        frozen.append((direction[0], direction[1]))
    return tuple(frozen)


def _masked_zero(X: Tensor, mask: Tensor) -> Tensor:
    """Set invalid token rows to zero without changing valid entries."""

    return X.masked_fill(~mask.unsqueeze(-1), 0)


def _pair_mask(node_mask: Tensor) -> Tensor:
    """An ordered pair is valid exactly when both endpoint nodes are valid."""

    return node_mask.unsqueeze(2) & node_mask.unsqueeze(1)


def attention_distribution_diagnostics(
    weights: Tensor,
    query_mask: Tensor,
    key_mask: Tensor,
) -> dict[str, float | int]:
    """Summarize a masked attention distribution without retaining its graph."""

    probabilities = weights.detach().float()
    entropy = -(
        probabilities
        * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1)
    maximum = probabilities.max(dim=-1).values
    valid_queries = query_mask.unsqueeze(1).expand_as(entropy)
    selected_entropy = entropy.masked_select(valid_queries)
    selected_maximum = maximum.masked_select(valid_queries)
    counts = key_mask.sum(dim=-1).to(dtype=entropy.dtype)
    denominator = counts.clamp_min(1).log()
    while denominator.ndim < entropy.ndim:
        denominator = denominator.unsqueeze(-1)
        counts = counts.unsqueeze(-1)
    normalized = torch.where(
        counts > 1,
        entropy / denominator.clamp_min(torch.finfo(entropy.dtype).eps),
        torch.zeros_like(entropy),
    ).masked_select(valid_queries)
    return {
        "valid_queries": int(selected_entropy.numel()),
        "entropy_mean": float(selected_entropy.mean().cpu()),
        "normalized_entropy_mean": float(normalized.mean().cpu()),
        "effective_keys_mean": float(selected_entropy.exp().mean().cpu()),
        "maximum_weight_mean": float(selected_maximum.mean().cpu()),
    }


def _check_tiled_inference(training: bool, recording: bool) -> None:
    if training or torch.is_grad_enabled():
        raise RuntimeError("Tiled attention requires eval mode and disabled gradients.")
    if recording:
        raise RuntimeError("Tiled attention does not collect attention diagnostics.")


class DenseMultiheadAttention(nn.Module):
    """Explicit dense multi-head attention for possibly different streams."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.record_attention_diagnostics = False
        self.last_attention_diagnostics: Optional[dict[str, float | int]] = None
        self._query_tiles: Optional[tuple[int, int]] = None

    def _split_heads(self, X: Tensor) -> Tensor:
        batch_size, length, _ = X.shape
        return X.reshape(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _attend(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        query_mask: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            ~key_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = attention_softmax(scores)
        if self.record_attention_diagnostics:
            self.last_attention_diagnostics = attention_distribution_diagnostics(
                weights, query_mask, key_mask
            )
        return torch.matmul(self.attention_dropout(weights), V)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor,
        key_mask: Tensor,
    ) -> Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must have shape [B, m, d].")
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes.")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key/value must have the same batch size.")
        if query.shape[-1] != self.d_model or key.shape[-1] != self.d_model:
            raise ValueError("Attention inputs must use d_model features.")
        if query_mask.shape != query.shape[:2]:
            raise ValueError("query_mask must match the query token axes.")
        if key_mask.shape != key.shape[:2]:
            raise ValueError("key_mask must match the key token axes.")
        if not bool(key_mask.any(dim=1).all()):
            raise ValueError("Every example needs at least one valid key token.")

        Q = self._split_heads(self.query(query))
        K = self._split_heads(self.key(key))
        V = self._split_heads(self.value(value))
        if self._query_tiles is None:
            context = self._attend(Q, K, V, query_mask, key_mask)
        else:
            _check_tiled_inference(self.training, self.record_attention_diagnostics)
            short_tile, long_tile = self._query_tiles
            # Vertex queries can have n^2 keys; pair queries have only n
            # keys when reading vertices and can use more rows per tile.
            tile = long_tile if query.shape[1] > key.shape[1] else short_tile
            context = torch.cat(
                [
                    self._attend(
                        Q[:, :, start:start + tile], K, V,
                        query_mask[:, start:start + tile], key_mask,
                    )
                    for start in range(0, Q.shape[2], tile)
                ],
                dim=2,
            )
        context = context.transpose(1, 2).reshape(
            query.shape[0], query.shape[1], self.d_model
        )
        return _masked_zero(self.output(context), query_mask)


class TriangularPairAttention(nn.Module):
    r"""Compose two oriented ordered-pair views over a shared index r.

    Head ``h`` reads :math:`Y^{\alpha_h}_{ir}` and :math:`Y^{\beta_h}_{rj}`.
    ``stored`` is the manuscript's right-arrow view :math:`Y`; ``transpose``
    is its left-arrow view :math:`Y^\top`. Orientations change
    index access only: every head retains its own Q/K/value projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        head_directions: Optional[Sequence[PairHeadDirection]] = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value_left = nn.Linear(d_model, d_model)
        self.value_right = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.record_attention_diagnostics = False
        self.last_attention_diagnostics: Optional[dict[str, float | int]] = None
        self._triangle_tiles: Optional[tuple[int, int]] = None
        if head_directions is None:
            head_directions = make_triangular_head_directions(
                num_heads, "chain_only"
            )
        self.head_directions = _validate_head_directions(
            head_directions, num_heads
        )
        # These masks are architectural metadata, not learned state.  Keeping
        # them non-persistent preserves strict loading of historical
        # chain-only checkpoints because parameter/state-dict keys are
        # unchanged.
        left_transposed = torch.tensor(
            [left == "transpose" for left, _ in self.head_directions],
            dtype=torch.bool,
        ).view(1, num_heads, 1, 1, 1)
        right_transposed = torch.tensor(
            [right == "transpose" for _, right in self.head_directions],
            dtype=torch.bool,
        ).view(1, num_heads, 1, 1, 1)
        self.register_buffer(
            "_left_transposed", left_transposed, persistent=False
        )
        self.register_buffer(
            "_right_transposed", right_transposed, persistent=False
        )

    def _split_heads(self, Y: Tensor) -> Tensor:
        batch_size, n, _, _ = Y.shape
        return Y.reshape(
            batch_size, n, n, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)

    @staticmethod
    def _oriented_view(states: Tensor, transposed_heads: Tensor) -> Tensor:
        """Select ``states[i,j]`` or ``states[j,i]`` head by head."""

        return torch.where(
            transposed_heads,
            states.transpose(2, 3),
            states,
        )

    def _compose(
        self,
        Q: Tensor,
        K: Tensor,
        V_left: Tensor,
        V_right: Tensor,
        node_mask: Tensor,
        feature_tile: Optional[int] = None,
    ) -> Tensor:
        # Every source-row tile retains the full bridge axis r.
        logits = torch.einsum("bhird,bhrjd->bhijr", Q, K) * self.scale
        logits = logits.masked_fill(
            ~node_mask[:, None, None, None, :], torch.finfo(logits.dtype).min
        )
        weights = attention_softmax(logits)
        if self.record_attention_diagnostics:
            self.last_attention_diagnostics = attention_distribution_diagnostics(
                weights, _pair_mask(node_mask), node_mask
            )
        weights = self.attention_dropout(weights)
        if feature_tile is None:
            return torch.einsum("bhijr,bhird,bhrjd->bhijd", weights, V_left, V_right)
        return torch.cat(
            [
                torch.einsum(
                    "bhijr,bhird,bhrjd->bhijd", weights,
                    V_left[..., start:start + feature_tile],
                    V_right[..., start:start + feature_tile],
                )
                for start in range(0, V_left.shape[-1], feature_tile)
            ],
            dim=-1,
        )

    def forward(self, Y: Tensor, node_mask: Tensor) -> Tensor:
        if Y.ndim != 4 or Y.shape[1] != Y.shape[2]:
            raise ValueError("Y must have shape [B, n, n, d_model].")
        if Y.shape[-1] != self.d_model:
            raise ValueError("Y must use d_model pair features.")
        if node_mask.shape != Y.shape[:2]:
            raise ValueError("node_mask must have shape [B, n].")
        if not bool(node_mask.any(dim=1).all()):
            raise ValueError("Every example needs at least one valid node.")

        # Project once, then transpose pair indices only in the heads assigned
        # to the transposed view.  Feature-wise linear maps commute with this
        # transposition, so this is exactly Q(Y^T), K(Y^T), ... without a
        # second projection or any additional parameters.
        Q = self._oriented_view(
            self._split_heads(self.query(Y)), self._left_transposed
        )
        K = self._oriented_view(
            self._split_heads(self.key(Y)), self._right_transposed
        )
        V_left = self._oriented_view(
            self._split_heads(self.value_left(Y)), self._left_transposed
        )
        V_right = self._oriented_view(
            self._split_heads(self.value_right(Y)), self._right_transposed
        )

        if self._triangle_tiles is None:
            context = self._compose(Q, K, V_left, V_right, node_mask)
        else:
            _check_tiled_inference(self.training, self.record_attention_diagnostics)
            source_tile, feature_tile = self._triangle_tiles
            context = torch.cat(
                [
                    self._compose(
                        Q[:, :, start:start + source_tile], K,
                        V_left[:, :, start:start + source_tile], V_right,
                        node_mask, feature_tile,
                    )
                    for start in range(0, Q.shape[2], source_tile)
                ],
                dim=2,
            )
        context = context.permute(0, 2, 3, 1, 4).reshape(
            Y.shape[0], Y.shape[1], Y.shape[2], self.d_model
        )
        return _masked_zero(self.output(context), _pair_mask(node_mask))


class AlignedPointwiseResidual(nn.Module):
    r"""Incidence-aware pre-LayerNorm MLPs with identity residual skips.

    Vertices receive their diagonal pair, while pair ``(i,j)`` receives the
    ordered endpoints and the structural diagonal indicator.  The two MLPs
    are independent.  Two instances are used per EVT layer, one after each
    attention phase, so their weights are independent as well.
    """

    def __init__(
        self,
        d_model: int,
        feedforward_dim: int,
        dropout: float,
        theory_mode: bool = False,
    ) -> None:
        super().__init__()
        normalization = nn.Identity if theory_mode else nn.LayerNorm
        activation = nn.ReLU if theory_mode else nn.GELU
        self.vertex_norm = normalization(2 * d_model)
        self.pair_norm = normalization(3 * d_model + 1)
        self.vertex_mlp = nn.Sequential(
            nn.Linear(2 * d_model, feedforward_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * d_model + 1, feedforward_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.vertex_dropout = nn.Dropout(dropout)
        self.pair_dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: Tensor,
        Y: Tensor,
        node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, n, d_model = X.shape
        pair_mask = _pair_mask(node_mask)
        diagonal = torch.arange(n, device=X.device)
        pair_diagonal = Y[:, diagonal, diagonal, :]
        vertex_input = torch.cat((X, pair_diagonal), dim=-1)

        source = X.unsqueeze(2).expand(-1, -1, n, -1)
        target = X.unsqueeze(1).expand(-1, n, -1, -1)
        diagonal_indicator = torch.eye(
            n, dtype=X.dtype, device=X.device
        ).view(1, n, n, 1).expand(batch_size, -1, -1, -1)
        pair_input = torch.cat((Y, source, target, diagonal_indicator), dim=-1)

        X_next = X + self.vertex_dropout(
            self.vertex_mlp(self.vertex_norm(vertex_input))
        )
        Y_next = Y + self.pair_dropout(
            self.pair_mlp(self.pair_norm(pair_input))
        )
        if X_next.shape != (batch_size, n, d_model):
            raise RuntimeError("Aligned vertex MLP returned an invalid shape.")
        return _masked_zero(X_next, node_mask), _masked_zero(Y_next, pair_mask)


class EdgeVertexTransformerLayer(nn.Module):
    """One intra-stream-then-cross-stream EVT layer on embedded states.

    Pass vertex states ``X`` of shape ``[B, n, d_model]``, ordered-pair states
    ``Y`` of shape ``[B, n, n, d_model]``, and a boolean vertex mask of shape
    ``[B, n]``. The layer returns updated vertex and pair states of the same
    shapes.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        theory_mode: bool = False,
        pair_attention: str = "triangular",
        triangular_head_config: TriangularHeadConfig = "chain_only",
        triangular_residual_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        d_ff = 4 * d_model if feedforward_dim is None else feedforward_dim
        if d_ff < 1:
            raise ValueError("feedforward_dim must be positive.")
        if pair_attention not in {"triangular", "dense"}:
            raise ValueError("pair_attention must be 'triangular' or 'dense'.")
        if triangular_residual_init is not None and not math.isfinite(
            triangular_residual_init
        ):
            raise ValueError("triangular_residual_init must be finite when provided.")
        head_directions = make_triangular_head_directions(
            num_heads, triangular_head_config
        )
        if pair_attention == "dense" and triangular_head_config != "chain_only":
            raise ValueError(
                "A directional triangular_head_config requires "
                "pair_attention='triangular'."
            )
        normalization = nn.Identity if theory_mode else nn.LayerNorm
        self.pair_attention = pair_attention
        self.triangular_head_config = triangular_head_config
        self.vertex_intra_norm = normalization(d_model)
        self.pair_intra_norm = normalization(d_model)
        self.vertex_cross_norm = normalization(d_model)
        self.pair_cross_norm = normalization(d_model)
        self.vertex_self_attention = DenseMultiheadAttention(
            d_model, num_heads, attention_dropout
        )
        if pair_attention == "triangular":
            self.pair_triangular_attention = TriangularPairAttention(
                d_model,
                num_heads,
                attention_dropout,
                head_directions=head_directions,
            )
            if triangular_residual_init is None:
                self.register_parameter("triangular_residual_scale", None)
            else:
                self.triangular_residual_scale = nn.Parameter(
                    torch.tensor(float(triangular_residual_init))
                )
        else:
            # Ablated relational primitive.  Ordinary self-attention over the
            # n^2 pair tokens can mix *any* two pair states, but it has no
            # explicit bridge-indexed product across the two legs of a
            # composition: each score and each value reads a single pair token.
            # Models built with it therefore fall outside the envelope of the
            # exact two-strike kernel theorem.  With injective positions and
            # unrestricted capacity, however, this does not prove a strict
            # expressivity separation from 2-FWL.  The module is a finite-budget
            # mechanism control at matched depth, attention count, incidence
            # coupling and parameter budget.
            self.pair_dense_attention = DenseMultiheadAttention(
                d_model, num_heads, attention_dropout
            )
            self.register_parameter("triangular_residual_scale", None)
        self.pair_to_vertex_attention = DenseMultiheadAttention(
            d_model, num_heads, attention_dropout
        )
        self.vertex_to_pair_attention = DenseMultiheadAttention(
            d_model, num_heads, attention_dropout
        )
        self.vertex_intra_dropout = nn.Dropout(dropout)
        self.pair_intra_dropout = nn.Dropout(dropout)
        self.vertex_cross_dropout = nn.Dropout(dropout)
        self.pair_cross_dropout = nn.Dropout(dropout)
        self.intra_aligned_mlp = AlignedPointwiseResidual(
            d_model, d_ff, dropout, theory_mode=theory_mode
        )
        self.cross_aligned_mlp = AlignedPointwiseResidual(
            d_model, d_ff, dropout, theory_mode=theory_mode
        )

    def _pair_intra_update(
        self,
        Y_intra: Tensor,
        node_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        """Update the pair stream from itself, with the selected primitive."""

        if self.pair_attention == "triangular":
            update = self.pair_triangular_attention(Y_intra, node_mask)
            if self.triangular_residual_scale is not None:
                update = self.triangular_residual_scale * update
            return update
        batch_size, n, _, d_model = Y_intra.shape
        flat_Y = Y_intra.reshape(batch_size, n * n, d_model)
        flat_pair_mask = pair_mask.reshape(batch_size, n * n)
        update = self.pair_dense_attention(
            flat_Y, flat_Y, flat_Y, flat_pair_mask, flat_pair_mask
        )
        return update.reshape(batch_size, n, n, d_model)

    def forward(
        self,
        X: Tensor,
        Y: Tensor,
        node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Update the embedded vertex and pair states and return both tensors."""

        batch_size, n, _ = X.shape
        pair_mask = _pair_mask(node_mask)

        # Phase 1: parallel within-stream attention, then one aligned MLP pair.
        X_intra = _masked_zero(self.vertex_intra_norm(X), node_mask)
        Y_intra = _masked_zero(self.pair_intra_norm(Y), pair_mask)
        bar_X = X + self.vertex_intra_dropout(
            self.vertex_self_attention(
                X_intra, X_intra, X_intra, node_mask, node_mask
            )
        )
        bar_Y = Y + self.pair_intra_dropout(
            self._pair_intra_update(Y_intra, node_mask, pair_mask)
        )
        bar_X = _masked_zero(bar_X, node_mask)
        bar_Y = _masked_zero(bar_Y, pair_mask)
        X_half, Y_half = self.intra_aligned_mlp(bar_X, bar_Y, node_mask)

        # Parallel cross-stream stage: both updates read the same half-step.
        flat_Y_half = Y_half.reshape(batch_size, n * n, Y_half.shape[-1])
        flat_pair_mask = pair_mask.reshape(batch_size, n * n)
        X_cross = _masked_zero(self.vertex_cross_norm(X_half), node_mask)
        Y_cross = _masked_zero(self.pair_cross_norm(Y_half), pair_mask)
        flat_Y_cross = Y_cross.reshape(batch_size, n * n, Y_cross.shape[-1])

        # Phase 2: both cross attentions read the same half-step states.
        bar_X_half = X_half + self.vertex_cross_dropout(
            self.pair_to_vertex_attention(
                X_cross,
                flat_Y_cross,
                flat_Y_cross,
                node_mask,
                flat_pair_mask,
            )
        )
        flat_bar_Y_half = flat_Y_half + self.pair_cross_dropout(
            self.vertex_to_pair_attention(
                flat_Y_cross,
                X_cross,
                X_cross,
                flat_pair_mask,
                node_mask,
            )
        )
        bar_X_half = _masked_zero(bar_X_half, node_mask)
        bar_Y_half = _masked_zero(
            flat_bar_Y_half.reshape(batch_size, n, n, Y.shape[-1]),
            pair_mask,
        )
        return self.cross_aligned_mlp(bar_X_half, bar_Y_half, node_mask)


class EdgeVertexTransformer(nn.Module):
    r"""Stacked path-independent graph backbone :math:`(X,Y)\mapsto(X',Y')`.

    ``pair_attention='triangular'`` is the frozen architecture and the only
    setting covered by the exact :math:`2`-FWL kernel results.
    ``pair_attention='dense'`` replaces the triangular primitive by ordinary
    self-attention over the :math:`n^2` pair tokens.  It is a finite-budget
    mechanism ablation outside the exact kernel theorem; no strict asymptotic
    expressivity ordering is asserted.

    ``triangular_head_config`` fixes how the common multi-head width is split
    among stored/transposed bridge geometries.  The default ``chain_only``
    preserves historical checkpoints; ``balanced`` and ``dag_oriented`` are
    parameter-matched directional controls.
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
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        theory_mode: bool = False,
        pair_attention: str = "triangular",
        triangular_head_config: TriangularHeadConfig = "chain_only",
        pair_initialization: PairInitialization = "separate",
        triangular_residual_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("EdgeVertexTransformer requires at least one layer.")
        if d_h % num_heads != 0:
            raise ValueError("d_h must be divisible by num_heads.")
        if pair_attention not in {"triangular", "dense"}:
            raise ValueError("pair_attention must be 'triangular' or 'dense'.")
        if pair_initialization not in {"separate", "endpoint_lift"}:
            raise ValueError(
                "pair_initialization must be 'separate' or 'endpoint_lift'."
            )
        make_triangular_head_directions(num_heads, triangular_head_config)
        if pair_attention == "dense" and triangular_head_config != "chain_only":
            raise ValueError(
                "A directional triangular_head_config requires "
                "pair_attention='triangular'."
            )
        if theory_mode and (dropout != 0.0 or attention_dropout != 0.0):
            raise ValueError("theory_mode requires zero dropout.")
        if theory_mode and (
            node_projection == "mlp" or pair_projection == "mlp"
        ):
            raise ValueError("theory_mode requires affine input projections.")
        self.d_x = d_x
        self.d_y = d_y
        self.d_h = d_h
        self.theory_mode = theory_mode
        self.pair_attention = pair_attention
        self.triangular_head_config = triangular_head_config
        self.pair_initialization = pair_initialization
        self.node_projection = make_projection(d_x, d_h, node_projection)
        pair_input_dim = d_y if pair_initialization == "separate" else d_y + 2 * d_x + 1
        self.pair_projection = make_projection(
            pair_input_dim,
            d_h,
            pair_projection,
        )
        self.layers = nn.ModuleList(
            [
                EdgeVertexTransformerLayer(
                    d_model=d_h,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    theory_mode=theory_mode,
                    pair_attention=pair_attention,
                    triangular_head_config=triangular_head_config,
                    triangular_residual_init=triangular_residual_init,
                )
                for _ in range(num_layers)
            ]
        )

    def enable_tiled_attention(
        self,
        *,
        source_tile: int = 8,
        feature_tile: int = 2,
        short_query_tile: int = 32,
        long_query_tile: int = 1024,
    ) -> None:
        """Tile attention for inference without changing model parameters.

        Triangular attention splits source rows and value channels. Dense
        attention splits query rows. Each softmax retains all valid keys
        or bridges, so outputs agree up to floating-point rounding. Call
        the model in eval mode with gradients and diagnostics disabled.
        """
        sizes = (source_tile, feature_tile, short_query_tile, long_query_tile)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Attention tile sizes must be positive integers.")
        for attention in self.layers.modules():
            if isinstance(attention, TriangularPairAttention):
                attention._triangle_tiles = (source_tile, feature_tile)
            elif isinstance(attention, DenseMultiheadAttention):
                attention._query_tiles = (short_query_tile, long_query_tile)

    def _validate_inputs(self, X: Tensor, Y: Tensor, node_mask: Tensor) -> None:
        if X.ndim != 3:
            raise ValueError("X must have shape [B, n, d_x].")
        if Y.ndim != 4:
            raise ValueError("Y must have shape [B, n, n, d_y].")
        batch_size, n, d_x = X.shape
        if Y.shape[:3] != (batch_size, n, n):
            raise ValueError("Y must have shape [B, n, n, d_y].")
        if d_x != self.d_x or Y.shape[-1] != self.d_y:
            raise ValueError("X or Y has a feature dimension different from the model.")
        if node_mask.shape != (batch_size, n) or node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean with shape [B, n].")
        if not X.is_floating_point() or not Y.is_floating_point():
            raise ValueError("X and Y must be floating point tensors.")
        if X.device != Y.device or X.device != node_mask.device:
            raise ValueError("X, Y, and node_mask must be on the same device.")
        if not bool(node_mask.any(dim=1).all()):
            raise ValueError("Every example needs at least one valid graph node.")

    def initialize_inputs(
        self, X: Tensor, Y: Tensor, node_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Embed atomic vertex and ordered-pair labels before EVT layers."""

        pair_mask = _pair_mask(node_mask)
        X_hidden = _masked_zero(self.node_projection(X), node_mask)
        pair_input = Y
        if self.pair_initialization == "endpoint_lift":
            batch_size, n, _ = X.shape
            source = X.unsqueeze(2).expand(-1, -1, n, -1)
            target = X.unsqueeze(1).expand(-1, n, -1, -1)
            diagonal = torch.eye(n, dtype=X.dtype, device=X.device)
            diagonal = diagonal.view(1, n, n, 1).expand(
                batch_size, -1, -1, -1
            )
            pair_input = torch.cat((Y, source, target, diagonal), dim=-1)
        Y_hidden = _masked_zero(self.pair_projection(pair_input), pair_mask)
        return X_hidden, Y_hidden

    def encode_with_tape(
        self, X: Tensor, Y: Tensor, node_mask: Tensor
    ) -> Tuple[EdgeVertexOutput, Tuple[Tensor, ...]]:
        """Return the terminal graph state and the vertex tape read by V3.

        The tape is ``(X^1, ..., X^L)``: entry ``ell`` is the state after
        graph layer ``ell``. It is path independent and can therefore be
        cached once per immutable graph and reused by every path query. The
        Aligned-EVT retains the initial state ``X^0`` separately so it can set
        ``Z^0 = X^0 o P`` without redundantly injecting it at its first depth.
        """

        self._validate_inputs(X, Y, node_mask)
        X_hidden, Y_hidden = self.initialize_inputs(X, Y, node_mask)
        return self.encode_initialized_with_tape(X_hidden, Y_hidden, node_mask)

    def encode_initialized_with_tape(
        self, X_hidden: Tensor, Y_hidden: Tensor, node_mask: Tensor
    ) -> Tuple[EdgeVertexOutput, Tuple[Tensor, ...]]:
        """Encode already lifted states and return ``(X^1, ..., X^L)``."""

        vertex_tape = []
        for layer in self.layers:
            X_hidden, Y_hidden = layer(X_hidden, Y_hidden, node_mask)
            vertex_tape.append(X_hidden)
        return (
            EdgeVertexOutput(X=X_hidden, Y=Y_hidden),
            tuple(vertex_tape),
        )

    def forward(self, X: Tensor, Y: Tensor, node_mask: Tensor) -> EdgeVertexOutput:
        """Project graph features and encode vertices and ordered pairs."""

        output, _ = self.encode_with_tape(X, Y, node_mask)
        return output
