"""Shared projections and task heads for path encoders."""

from typing import Optional

import torch
from torch import Tensor, nn

from path_augmented_transformer.data_types import (
    PathEncoderOutput,
    ScalarTaskOutput,
    StructuredTaskOutput,
    junction_mask_from_path_mask,
    masked_path_mean,
)


def attention_softmax(scores: Tensor, dim: int = -1) -> Tensor:
    """Apply a stable softmax without discarding float64 precision.

    Half-precision scores are promoted to float32 for numerical stability;
    float32 and float64 inputs retain their native accumulation precision.
    """

    compute_dtype = (
        scores.dtype if scores.dtype in {torch.float32, torch.float64}
        else torch.float32
    )
    return torch.softmax(scores.to(compute_dtype), dim=dim).to(scores.dtype)


def make_projection(d_in: int, d_out: int, kind: str) -> nn.Module:
    r"""Build the pointwise map :math:`\phi_X`."""

    if kind == "identity":
        if d_in != d_out:
            raise ValueError("identity projection requires d_x == d_h.")
        return nn.Identity()
    if kind == "linear":
        return nn.Linear(d_in, d_out)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
    raise ValueError("projection must be one of: 'identity', 'linear', 'mlp'.")


def make_mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class JunctionTokenBuilder(nn.Module):
    """Derive final path-junction tokens from adjacent occurrence tokens.

    These path-only models have no pair stream, so this implements
    ``J_t = phi_J(Z_t, Z_{t+1}, 0)`` with the zero argument omitted.
    """

    def __init__(self, d_h: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = make_mlp(2 * d_h, d_h, d_h, dropout)

    def forward(self, Z: Tensor, path_mask: Tensor) -> Tensor:
        if Z.shape[1] < 2:
            return Z.new_zeros((Z.shape[0], 0, Z.shape[2]))
        J = self.network(torch.cat((Z[:, :-1], Z[:, 1:]), dim=-1))
        junction_mask = junction_mask_from_path_mask(path_mask)
        return J.masked_fill(~junction_mask.unsqueeze(-1), 0)


class ScalarPathHead(nn.Module):
    """Mean-pool encoder tokens and predict a scalar or short vector."""

    def __init__(
        self,
        d_h: int,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_hidden = d_h if hidden_dim is None else hidden_dim
        self.network = make_mlp(d_h, d_hidden, output_dim, dropout)

    def forward(self, encoding: PathEncoderOutput, path_mask: Tensor) -> ScalarTaskOutput:
        """Predict from the mean of the unmasked path vertex states."""

        u = masked_path_mean(encoding.Z, path_mask)
        return ScalarTaskOutput(u=u, prediction=self.network(u))


class StructuredPathHead(nn.Module):
    """Build optional junction tokens and make aligned predictions.

    Path-only encoders derive ``J_t`` from adjacent occurrence tokens.  An
    encoder such as Strike Twice may instead expose a complete occurrence-pair
    array ``R``; in that case ``J_t = R[t,t+1]`` is used directly.
    """

    def __init__(
        self,
        d_h: int,
        node_output_dim: int = 1,
        junction_output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_hidden = d_h if hidden_dim is None else hidden_dim
        self.junction_builder = JunctionTokenBuilder(d_h=d_h, dropout=dropout)
        self.node_head = make_mlp(d_h, d_hidden, node_output_dim, dropout)
        self.junction_head = make_mlp(d_h, d_hidden, junction_output_dim, dropout)

    def forward(
        self,
        encoding: PathEncoderOutput,
        path_mask: Tensor,
    ) -> StructuredTaskOutput:
        if encoding.R is None:
            J = self.junction_builder(encoding.Z, path_mask)
        else:
            if encoding.R.ndim != 4:
                raise ValueError("R must have shape [B, k, k, d_h].")
            batch_size, path_length, _, d_h = encoding.R.shape
            if encoding.R.shape[:3] != (batch_size, path_length, path_length):
                raise ValueError("R must have shape [B, k, k, d_h].")
            if path_length != encoding.Z.shape[1] or d_h != encoding.Z.shape[2]:
                raise ValueError("R and Z must use the same path length and hidden width.")
            positions = torch.arange(max(0, path_length - 1), device=encoding.R.device)
            J = encoding.R[:, positions, positions + 1]
            junction_mask = junction_mask_from_path_mask(path_mask)
            J = J.masked_fill(~junction_mask.unsqueeze(-1), 0)
        node_logits = self.node_head(encoding.Z)
        junction_logits = self.junction_head(J)

        node_logits = node_logits.masked_fill(~path_mask.unsqueeze(-1), 0)
        junction_mask = junction_mask_from_path_mask(path_mask)
        junction_logits = junction_logits.masked_fill(~junction_mask.unsqueeze(-1), 0)
        return StructuredTaskOutput(
            J=J,
            node_logits=node_logits,
            junction_logits=junction_logits,
        )
