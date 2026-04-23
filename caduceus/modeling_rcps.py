"""RC-equivariant building blocks: embedding, generic wrapper, Mamba block,
add-norm wrapper, LM head. Treats forward and RC strands symmetrically."""

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# NOTE: mamba_ssm requires a CUDA GPU and must be installed from source
# see https://github.com/state-spaces/mamba
try:
    from mamba_ssm.ops.triton.layernorm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn,
    )  # mamba v1
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import (
            RMSNorm,
            layer_norm_fn,
            rms_norm_fn,
        )  # mamba v2
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class RCPSEmbedding(nn.Module):
    """Token embedding with RC equivariance; output is [fwd || rc], dim = 2 * d_model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        complement_map: dict,
        **factory_kwargs,
    ):
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long),
        )
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

    @property
    def weight(self) -> torch.Tensor:
        """Embedding weight matrix."""
        return self.embedding.weight

    def set_weight(self, value: torch.Tensor) -> None:
        self.embedding.weight = value

    def rc(self, x: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            self.complement_map.unsqueeze(0).expand(x.shape[0], -1),
            dim=1,
            index=torch.flip(x, dims=[-1]),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        fwd_out = self.embedding(input_ids)
        rc_out = torch.flip(self.embedding(self.rc(input_ids)), dims=[-2, -1])
        return torch.cat([fwd_out, rc_out], dim=-1)


class RCPSWrapper(nn.Module):
    """Wrap any nn.Module to make it RC-equivariant over concatenated [fwd || rc] tensors."""

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    @staticmethod
    def rc(x: torch.Tensor) -> torch.Tensor:
        """Flip both sequence (dim=-2) and channel (dim=-1) dimensions."""
        return torch.flip(x, dims=[-2, -1])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        n_channels = x.shape[-1]
        fwd_out = self.submodule(x[..., : n_channels // 2], **kwargs)
        rc_out = self.submodule(self.rc(x[..., n_channels // 2 :]), **kwargs)
        return torch.cat([fwd_out, self.rc(rc_out)], dim=-1)


class RCPSMambaBlock(nn.Module):
    """RC-equivariant residual Mamba block; drop-in replacement when rcps=True."""

    def __init__(
        self,
        dim: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = RCPSWrapper(mixer_cls(dim))
        norm_f = norm_cls(dim)
        self.norm = norm_f if fused_add_norm else RCPSAddNormWrapper(norm_f)
        if self.fused_add_norm:
            if RMSNorm is None:
                raise RuntimeError(
                    "fused_add_norm=True requires RMSNorm from mamba_ssm. "
                    "Install mamba_ssm from source: https://github.com/state-spaces/mamba"
                )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        token_ids=None,
    ):
        if not self.fused_add_norm:
            hidden_states_norm, residual = self.norm(
                hidden_states, residual=residual, prenorm=True
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            raise NotImplementedError(
                "fused_add_norm path is not yet implemented for RCPSMambaBlock. "
                "Set fused_add_norm=False in CaduceusConfig."
            )

        hidden_states = self.mixer(
            hidden_states_norm,
            inference_params=inference_params,
            token_ids=token_ids,
        )
        return hidden_states, residual


class RCPSAddNormWrapper(RCPSWrapper):
    """RC-equivariant add-norm layer; applies residual + norm independently per strand."""

    def __init__(self, submodule: nn.Module):
        super().__init__(submodule)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
    ):
        n_channels = x.shape[-1]
        if residual is None:
            residual = x
            x_fwd = self.submodule(
                x[..., : n_channels // 2].to(dtype=self.submodule.weight.dtype)
            )
            x_rc = self.submodule(
                self.rc(x[..., n_channels // 2 :]).to(dtype=self.submodule.weight.dtype)
            )
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)
        else:
            residual_fwd = x[..., : n_channels // 2] + residual[..., : n_channels // 2]
            x_fwd = self.submodule(residual_fwd.to(dtype=self.submodule.weight.dtype))

            residual_rc = self.rc(x[..., n_channels // 2 :]) + self.rc(
                residual[..., n_channels // 2 :]
            )
            x_rc = self.submodule(residual_rc.to(dtype=self.submodule.weight.dtype))

            residual = torch.cat([residual_fwd, self.rc(residual_rc)], dim=-1)
            x = torch.cat([x_fwd, self.rc(x_rc)], dim=-1)

        return x if not prenorm else (x, residual)

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ):
        return self.submodule.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


class RCPSLMHead(nn.Module):
    """LM head for RCPS inputs (dim = 2 * true_dim); averages logits from both strands."""

    def __init__(
        self,
        true_dim: int,
        vocab_size: int,
        complement_map: dict,
        **factory_kwargs,
    ):
        super().__init__()
        self.register_buffer(
            "complement_map",
            torch.tensor(list(OrderedDict(complement_map).values()), dtype=torch.long),
        )
        self.true_dim = true_dim
        self.lm_head = nn.Linear(true_dim, vocab_size, bias=False, **factory_kwargs)

    @property
    def weight(self) -> torch.Tensor:
        """LM head weight matrix."""
        return self.lm_head.weight

    def set_weight(self, value: torch.Tensor) -> None:
        self.lm_head.weight = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_channels = x.shape[-1]
        if n_channels != 2 * self.true_dim:
            raise ValueError(
                f"Input last dimension must be 2 * true_dim = {2 * self.true_dim}, "
                f"got {n_channels}."
            )
        fwd_logits = F.linear(
            x[..., : n_channels // 2], self.weight, bias=self.lm_head.bias
        )
        rc_logits = F.linear(
            torch.flip(x[..., n_channels // 2 :], dims=[-1]),
            self.weight[self.complement_map, :],
            bias=self.lm_head.bias,
        )
        return fwd_logits + rc_logits
