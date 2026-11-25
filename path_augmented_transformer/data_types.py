"""Typed batches, encoder states, and task-head outputs."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gather_path_nodes(X: Tensor, p: Tensor, path_mask: Tensor) -> Tensor:
    """Return the ordered gather ``X o p`` and zero padded path positions.

    ``p`` may contain any sentinel value at padded positions. Only indices for
    which ``path_mask`` is true are required to belong to ``[0, n)``.
    """

    _require(X.ndim == 3, "X must have shape [B, n, d_x].")
    _require(p.ndim == 2, "p must have shape [B, k].")
    _require(path_mask.ndim == 2, "path_mask must have shape [B, k].")
    _require(p.shape == path_mask.shape, "p and path_mask must have equal shapes.")
    _require(X.shape[0] == p.shape[0], "X and p must have the same batch size.")
    _require(p.dtype == torch.long, "p must use torch.long indices.")
    _require(path_mask.dtype == torch.bool, "path_mask must be boolean.")

    n = X.shape[1]
    valid_indices = p[path_mask]
    if valid_indices.numel() > 0:
        _require(bool((valid_indices >= 0).all()), "Valid path indices must be non-negative.")
        _require(bool((valid_indices < n).all()), "A valid path index exceeds n - 1.")

    safe_p = p.masked_fill(~path_mask, 0)
    gather_index = safe_p.unsqueeze(-1).expand(-1, -1, X.shape[-1])
    gathered = torch.gather(X, dim=1, index=gather_index)
    return gathered.masked_fill(~path_mask.unsqueeze(-1), 0)


def gather_path_pairs(Y: Tensor, p: Tensor, path_mask: Tensor) -> Tensor:
    r"""Return the ordered pair pullback ``Y[p_s, p_t]``.

    The returned array has shape ``[B, k, k, d_y]``.  Padded occurrence
    pairs are exactly zero, and sentinel values in ``p`` are never indexed.
    Repeated path vertices are preserved as distinct occurrence rows.
    """

    _require(Y.ndim == 4, "Y must have shape [B, n, n, d_y].")
    _require(Y.shape[1] == Y.shape[2], "Y must have square pair axes.")
    _require(p.ndim == 2, "p must have shape [B, k].")
    _require(path_mask.shape == p.shape, "path_mask must match p.")
    _require(Y.shape[0] == p.shape[0], "Y and p must have the same batch size.")
    _require(p.dtype == torch.long, "p must use torch.long indices.")
    _require(path_mask.dtype == torch.bool, "path_mask must be boolean.")

    batch_size, n, _, _ = Y.shape
    valid_indices = p[path_mask]
    if valid_indices.numel() > 0:
        _require(bool((valid_indices >= 0).all()), "Valid path indices must be non-negative.")
        _require(bool((valid_indices < n).all()), "A valid path index exceeds n - 1.")

    safe_p = p.masked_fill(~path_mask, 0)
    k = p.shape[1]
    batch = torch.arange(batch_size, device=Y.device).view(batch_size, 1, 1)
    source = safe_p.unsqueeze(2).expand(-1, -1, k)
    target = safe_p.unsqueeze(1).expand(-1, k, -1)
    gathered = Y[batch, source, target]
    pair_mask = path_mask.unsqueeze(2) & path_mask.unsqueeze(1)
    return gathered.masked_fill(~pair_mask.unsqueeze(-1), 0)


def junction_mask_from_path_mask(path_mask: Tensor) -> Tensor:
    """A junction is valid exactly when both adjacent path positions exist."""

    _require(path_mask.ndim == 2, "path_mask must have shape [B, k].")
    _require(path_mask.dtype == torch.bool, "path_mask must be boolean.")
    return path_mask[:, :-1] & path_mask[:, 1:]


def masked_path_mean(Z: Tensor, path_mask: Tensor) -> Tensor:
    """Compute the exact arithmetic mean over valid path positions."""

    _require(Z.ndim == 3, "Z must have shape [B, k, d_h].")
    _require(path_mask.shape == Z.shape[:2], "path_mask must match the first two axes of Z.")
    _require(path_mask.dtype == torch.bool, "path_mask must be boolean.")
    lengths = path_mask.sum(dim=1)
    _require(bool((lengths > 0).all()), "Every example must contain at least one valid path token.")
    weights = path_mask.unsqueeze(-1).to(dtype=Z.dtype)
    return (Z * weights).sum(dim=1) / lengths.unsqueeze(-1).to(dtype=Z.dtype)


@dataclass(frozen=True)
class GraphPathBatch:
    """A padded batch of graph/path examples.

    Shapes follow the paper's ``(X, Y, P)`` convention. In code, the path is
    named ``p`` and padded positions conventionally contain ``-1``.
    """

    X: Tensor  # [B, n, d_x]
    Y: Tensor  # [B, n, n, d_y]
    p: Tensor  # [B, k]
    node_mask: Tensor  # [B, n]
    path_mask: Tensor  # [B, k]
    # Optional batching metadata. Equal values identify repeated query rows of
    # the same immutable graph, allowing graph encoders to run only once.
    graph_index: Optional[Tensor] = None  # [B]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.X.shape[0], self.X.shape[1], self.p.shape[1]

    @property
    def path_lengths(self) -> Tensor:
        return self.path_mask.sum(dim=1)

    @property
    def junction_mask(self) -> Tensor:
        return junction_mask_from_path_mask(self.path_mask)

    def gather_path_nodes(self) -> Tensor:
        return gather_path_nodes(self.X, self.p, self.path_mask)

    def gather_path_pairs(self) -> Tensor:
        return gather_path_pairs(self.Y, self.p, self.path_mask)

    def validate(self) -> None:
        _require(self.X.ndim == 3, "X must have shape [B, n, d_x].")
        _require(self.Y.ndim == 4, "Y must have shape [B, n, n, d_y].")
        _require(self.p.ndim == 2, "p must have shape [B, k].")
        _require(self.node_mask.ndim == 2, "node_mask must have shape [B, n].")
        _require(self.path_mask.ndim == 2, "path_mask must have shape [B, k].")

        batch_size, n, _ = self.X.shape
        _require(self.Y.shape[:3] == (batch_size, n, n), "Y axes must be [B, n, n, d_y].")
        _require(self.p.shape[0] == batch_size, "p must have the same batch size as X.")
        _require(self.node_mask.shape == (batch_size, n), "node_mask must have shape [B, n].")
        _require(self.path_mask.shape == self.p.shape, "path_mask must have the same shape as p.")
        _require(self.p.dtype == torch.long, "p must use torch.long indices.")
        _require(self.node_mask.dtype == torch.bool, "node_mask must be boolean.")
        _require(self.path_mask.dtype == torch.bool, "path_mask must be boolean.")
        if self.graph_index is not None:
            _require(
                self.graph_index.shape == (batch_size,),
                "graph_index must have shape [B].",
            )
            _require(
                self.graph_index.dtype == torch.long,
                "graph_index must use torch.long indices.",
            )
            _require(
                self.graph_index.device == self.X.device,
                "graph_index must be on the same device as X.",
            )
        _require(self.X.is_floating_point(), "X must be floating point.")
        _require(self.Y.is_floating_point(), "Y must be floating point.")
        _require(bool((self.path_lengths > 0).all()), "Every example needs at least one path token.")

        # Batches use prefix masks: valid positions followed only by padding.
        if self.path_mask.shape[1] > 1:
            has_hole = self.path_mask[:, 1:] & ~self.path_mask[:, :-1]
            _require(not bool(has_hole.any()), "path_mask must be left-aligned without holes.")

        # This also validates the numerical range of every unmasked path index.
        safe_p = self.p.masked_fill(~self.path_mask, 0)
        gather_path_nodes(self.X, self.p, self.path_mask)
        selected_nodes_exist = torch.gather(self.node_mask, dim=1, index=safe_p)
        _require(
            bool((selected_nodes_exist | ~self.path_mask).all()),
            "A valid path position references a padded graph node.",
        )

    def to(self, *args: object, **kwargs: object) -> "GraphPathBatch":
        """Return a batch with all tensors moved using ``Tensor.to``."""

        moved_X = self.X.to(*args, **kwargs)
        return GraphPathBatch(
            X=moved_X,
            Y=self.Y.to(*args, **kwargs),
            # A requested floating-point dtype applies only to feature arrays.
            p=self.p.to(device=moved_X.device),
            node_mask=self.node_mask.to(device=moved_X.device),
            path_mask=self.path_mask.to(device=moved_X.device),
            graph_index=(
                None
                if self.graph_index is None
                else self.graph_index.to(device=moved_X.device)
            ),
        )


@dataclass
class EdgeVertexOutput:
    r"""Contextualized graph states returned by the EVT backbone."""

    X: Tensor  # [B, n, d_h]
    Y: Tensor  # [B, n, n, d_h]


@dataclass
class PathEncoderOutput:
    r"""Typed encoder output corresponding to :math:`(X',Y',Z)` or
    :math:`(X',Y',Z,R)`.

    For the path-only controls, ``X`` is the shared pointwise projection of
    every input node and ``Y`` is passed through unchanged. Their path stream
    ``Z`` still reads only the gathered nodes ``X o P``. EVT encoders replace
    ``X`` and ``Y`` by contextualized graph states. Strike Twice additionally
    exposes the second-strike occurrence-pair states ``R`` so a structured
    task head can use them without changing the graph-level output fields.
    """

    X: Tensor  # [B, n, d_x_prime]
    Y: Tensor  # [B, n, n, d_y_prime]
    Z: Tensor  # [B, k, d_h]
    R: Optional[Tensor] = None  # [B, k, k, d_r], occurrence-pair states


@dataclass
class ScalarTaskOutput:
    """Output of a scalar task head; ``u`` is derived from encoder tokens."""

    u: Tensor  # [B, d_h]
    prediction: Tensor  # [B, d_scalar]


@dataclass
class StructuredTaskOutput:
    """Output of a task head aligned with path positions and junctions."""

    J: Tensor  # [B, k - 1, d_h]
    node_logits: Tensor  # [B, k, d_node]
    junction_logits: Tensor  # [B, k - 1, d_junction]
