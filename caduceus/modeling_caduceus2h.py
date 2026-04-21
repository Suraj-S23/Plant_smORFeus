"""Caduceus model architecture for per-nucleotide plant genome annotation.

This module implements the full bidirectional Mamba backbone (CaduceusMixerModel)
and its downstream classification heads (Caduceus). It also contains the
helper modules that construct individual Mamba blocks (create_block,
BiMambaWrapper) and the embedding layer (CaduceusEmbeddings).

The architecture is used by plant_train.py as the backbone of
PlantAnnotationModel: it accepts tokenised DNA chunks and returns per-position
hidden states that are then passed to the convolutional decoder.

Key classes:
    BiMambaWrapper          -- Forward + RC Mamba with configurable fusion
    CaduceusEmbeddings      -- Token + optional frame-positional embeddings
    CaduceusMixerModel      -- Full backbone (embeddings + blocks + norm)
    CaduceusPreTrainedModel -- HF base class with weight init
    Caduceus                -- Backbone + annotation heads

NOTE: mamba_ssm requires a CUDA GPU and must be installed from source: see https://github.com/state-spaces/mamba
"""

import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

# NOTE: mamba_ssm requires a CUDA GPU and must be installed from source
# — see https://github.com/state-spaces/mamba
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError as e:
    raise ImportError(
        "mamba_ssm is required but could not be imported. "
        "This package requires a CUDA-capable GPU and must be installed from source. "
        "Please follow the installation instructions at: "
        "https://github.com/state-spaces/mamba\n"
        f"Original error: {e}"
    ) from e

try:  # mamba v1
    from mamba_ssm.modules.mamba_simple import Block
except ImportError:  # mamba v2
    from mamba_ssm.modules.block import Block

try:  # Triton layer-norm kernels (mamba v1 path)
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn  # noqa: F401
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn  # noqa: F401

from mamba_ssm.modules.mlp import GatedMLP

from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import (
    RCPSEmbedding,
    RCPSLMHead,
    RCPSMambaBlock,
    RCPSWrapper,
)


# ---------------------------------------------------------------------------
# Block factory
# ---------------------------------------------------------------------------

def create_block(
    d_model: int,
    ssm_cfg: Optional[dict] = None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = False,
    layer_idx: Optional[int] = None,
    bidirectional: bool = True,
    bidirectional_strategy: str = "add",
    bidirectional_weight_tie: bool = True,
    rcps: bool = False,
    device=None,
    dtype=None,
    complement_map: Optional[Dict[int, int]] = None,
) -> nn.Module:
    """Construct a single Mamba residual block (BiMambaWrapper inside Block or RCPSMambaBlock)."""
    if ssm_cfg is None:
        ssm_cfg = {}
    factory = {"device": device, "dtype": dtype}

    mixer_cls = partial(
        BiMambaWrapper,
        layer_idx=layer_idx,
        bidirectional=bidirectional,
        bidirectional_strategy=bidirectional_strategy,
        bidirectional_weight_tie=bidirectional_weight_tie,
        complement_map=complement_map,
        **ssm_cfg,
        **factory,
    )

    norm_cls = partial(RMSNorm if rms_norm else nn.LayerNorm, eps=norm_epsilon, **factory)
    block_cls = RCPSMambaBlock if rcps else Block

    block = block_cls(
        d_model,
        mixer_cls,
        mlp_cls=partial(GatedMLP, hidden_features=d_model * 2, **factory),
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ---------------------------------------------------------------------------
# Bidirectional Mamba wrapper
# ---------------------------------------------------------------------------

class BiMambaWrapper(nn.Module):
    """Two-strand Mamba (forward + reverse-complement) with configurable fusion."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        bidirectional: bool = True,
        bidirectional_strategy: str = "add",  # "add" | "gated" | "concat_linear"
        bidirectional_weight_tie: bool = True,
        complement_map: Optional[Dict[int, int]] = None,
        vocab_size: int = 65,
        **mamba_kwargs,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        # Complement look-up table (persistent=False — reconstructed from config)
        self.register_buffer(
            "complement_lut",
            torch.full((vocab_size,), 5, dtype=torch.long),
            persistent=False,
        )
        for k, v in (complement_map or {}).items():
            if k < vocab_size:
                self.complement_lut[k] = v

        self.mamba_fwd = Mamba(d_model=d_model, layer_idx=layer_idx, **mamba_kwargs)
        self.mamba_rev = (
            Mamba(d_model=d_model, layer_idx=layer_idx, **mamba_kwargs)
            if bidirectional
            else None
        )
        if bidirectional and bidirectional_weight_tie:
            self.mamba_rev.load_state_dict(self.mamba_fwd.state_dict())

        # Embedding handles — populated by the backbone via set_embeddings()
        self.word_embeddings: Optional[nn.Embedding] = None
        self.frame_embeddings: Optional[nn.Embedding] = None
        self.use_frame_pos: bool = True

        # Fusion modules (only instantiated for the relevant strategy)
        if self.bidirectional and self.bidirectional_strategy == "gated":
            # Scalar gate per token in [0, 1], computed from [fwd; rev]
            self.fuse_gate = nn.Linear(2 * d_model, 1)
        elif self.bidirectional and self.bidirectional_strategy == "concat_linear":
            # Concatenate then project back to d_model
            self.fuse_proj = nn.Linear(2 * d_model, d_model)

    def set_embeddings(
        self,
        word_embeddings: nn.Embedding,
        frame_embeddings: Optional[nn.Embedding],
        use_frame_pos: bool,
    ) -> None:
        """Store embedding references so the RC stream can look them up."""
        self.word_embeddings = word_embeddings
        self.frame_embeddings = frame_embeddings
        self.use_frame_pos = bool(use_frame_pos)

    def _embed_rc(self, rc_ids: torch.Tensor) -> torch.Tensor:
        """Embed reverse-complement token ids (frame-pos omitted on RC stream)."""
        if self.word_embeddings is None:
            raise RuntimeError(
                "word_embeddings is not set. Call set_embeddings() before forward()."
            )
        return self.word_embeddings(rc_ids)

    def reverse_complement(self, ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to their reverse complement via the complement LUT."""
        return self.complement_lut[torch.flip(ids, dims=[1])]

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        out_fwd = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if not self.bidirectional:
            return out_fwd

        rev_in = torch.flip(hidden_states, dims=[1])
        out_rev = self.mamba_rev(rev_in, inference_params=inference_params)
        out_rev = torch.flip(out_rev, dims=[1])

        if self.bidirectional_strategy == "add":
            return out_fwd + out_rev
        elif self.bidirectional_strategy == "gated":
            gate = torch.sigmoid(
                self.fuse_gate(torch.cat([out_fwd, out_rev], dim=-1))
            )  # [B, L, 1]
            return (1.0 - gate) * out_fwd + gate * out_rev
        elif self.bidirectional_strategy == "concat_linear":
            return self.fuse_proj(torch.cat([out_fwd, out_rev], dim=-1))
        else:
            return out_fwd + out_rev


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class CaduceusEmbeddings(nn.Module):
    """Token embeddings with optional modulo-3 frame-positional encoding.

    frame_pos_mode: "off" (none), "local" (pos % 3), "global" ((phase + pos) % 3).
    """

    def __init__(self, config: CaduceusConfig, **factory):
        super().__init__()
        if getattr(config, "vocab_size", 6) < 6:
            raise ValueError(
                "vocab_size must be >= 6 because PAD uses token id 5. "
                f"Got vocab_size={config.vocab_size}."
            )

        if getattr(config, "rcps", False):
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size,
                config.d_model,
                getattr(config, "complement_map", None),
                **factory,
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.d_model, **factory
            )

        self.use_frame_pos = bool(getattr(config, "use_frame_positional_encoding", True))
        self.frame_pos_mode = getattr(config, "frame_pos_mode", "global")
        if self.use_frame_pos:
            self.frame_embeddings = nn.Embedding(3, config.d_model, **factory)
        else:
            self.frame_embeddings = None

    def forward(
        self,
        ids: torch.Tensor,
        frame_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ids.dtype not in (torch.int64, torch.int32):
            ids = ids.long()
        x = self.word_embeddings(ids)  # [B, L, D]

        if self.use_frame_pos and self.frame_embeddings is not None:
            mode = self.frame_pos_mode
            B, L = ids.shape
            if mode == "off":
                pass
            elif mode == "local" or frame_phase is None:
                mod3 = (
                    (torch.arange(L, device=ids.device) % 3).long().unsqueeze(0).expand(B, L)
                )
                x = x + self.frame_embeddings(mod3)
            else:  # "global"
                offsets = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
                mod3 = ((frame_phase.view(-1, 1) + offsets) % 3).long()
                x = x + self.frame_embeddings(mod3)

        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class CaduceusMixerModel(nn.Module):
    """Bidirectional Mamba backbone: embeddings + stacked blocks + final norm."""

    def __init__(self, config: CaduceusConfig, **factory):
        super().__init__()
        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = CaduceusEmbeddings(config, **factory)
        # Propagate the frame-pos mode explicitly so it reflects the config
        self.embeddings.frame_pos_mode = getattr(config, "frame_pos_mode", "global")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    complement_map=config.complement_map,
                    **factory,
                )
                for i in range(config.n_layer)
            ]
        )

        # Give every BiMambaWrapper a reference to the shared embedding modules
        for layer in self.layers:
            mixer = getattr(layer, "mixer", None)
            if mixer is None:
                continue
            target = mixer.submodule if isinstance(mixer, RCPSWrapper) else mixer
            if hasattr(target, "set_embeddings"):
                fe = getattr(self.embeddings, "frame_embeddings", None)
                target.set_embeddings(
                    self.embeddings.word_embeddings, fe, self.embeddings.use_frame_pos
                )

        norm_cls = RMSNorm if config.rms_norm else nn.LayerNorm
        self.norm_final = norm_cls(config.d_model, eps=config.norm_epsilon, **factory)

    def forward(
        self,
        *,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        frame_phase: Optional[torch.Tensor] = None,
        cache_params=None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithNoAttention:
        if input_ids is not None:
            token_ids = input_ids
            hidden = self.embeddings(input_ids, frame_phase=frame_phase)
        elif inputs_embeds is not None:
            token_ids = None
            hidden = inputs_embeds
        else:
            raise ValueError("Provide either input_ids or inputs_embeds.")

        residual = None
        hiddens = [] if output_hidden_states else None

        for layer in self.layers:
            hidden, residual = layer(
                hidden,
                residual=residual,
                inference_params=cache_params,
                token_ids=token_ids,
            )
            if hiddens is not None:
                hiddens.append(hidden)

        hidden = self.norm_final((hidden + residual).to(self.norm_final.weight.dtype))
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden,
            hidden_states=hiddens,
        )


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

class CaduceusPreTrainedModel(PreTrainedModel):
    """HuggingFace base class with Caduceus-specific weight initialisation.

    Applies rescaled pre-norm residual initialisation to output projection
    weights, consistent with the smORFeus pre-training recipe.
    """

    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(self, module: nn.Module, initializer_range: float = 0.02, **kwargs):
        """Initialise linear/embedding weights; rescale out_proj and fc2 by 1/sqrt(n_layer)."""
        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg or {}
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_layer)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class Caduceus(CaduceusPreTrainedModel):
    """Caduceus backbone with three per-frame heads and one ORF-presence head."""

    config_class = CaduceusConfig
    base_model_prefix = "caduceus"

    def __init__(self, config: CaduceusConfig, **factory):
        super().__init__(config)
        self.backbone = CaduceusMixerModel(config, **factory)
        self.frame_heads = nn.ModuleList(
            [nn.Linear(config.d_model, 1, **factory) for _ in range(3)]
        )
        self.orf_head = nn.Linear(config.d_model, 1, **factory)
        self.post_init()

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        cache_params=None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        frame_phase: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithNoAttention:
        outputs = self.backbone(
            input_ids=input_ids,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            frame_phase=frame_phase,
        )
        hs = outputs.last_hidden_state  # [B, L, H]

        frame_logits = torch.stack(
            [head(hs).squeeze(-1) for head in self.frame_heads], dim=-1
        )  # [B, L, 3]

        orf_logits = self.orf_head(hs).squeeze(-1)  # [B, L]

        out = BaseModelOutputWithNoAttention(last_hidden_state=frame_logits)
        out.frame_logits = frame_logits
        out.orf_logits = orf_logits
        if output_hidden_states:
            out.hidden_states = outputs.hidden_states
        return out


# ---------------------------------------------------------------------------
# HF task heads (kept for pre-training / transfer compatibility)
# ---------------------------------------------------------------------------

class CaduceusForMaskedLM(CaduceusPreTrainedModel):
    """Caduceus masked-LM head used during pre-training; weights loaded by PlantTrainer."""

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,
                vocab_size=self.config.vocab_size,
                true_dim=config.d_model,
                dtype=dtype,
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,
                bias=False,
                **factory_kwargs,
            )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the word embedding module."""
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if self.config.rcps:
            raise NotImplementedError(
                "Setting input embeddings is not supported in RCPS mode."
            )
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if self.config.rcps:
            raise NotImplementedError(
                "Setting output embeddings is not supported in RCPS mode."
            )
        self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        """Tie input and output embedding weights, accounting for RCPS."""
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self) -> nn.Module:
        """Return the backbone (decoder) module."""
        return self.caduceus

    def set_decoder(self, decoder: nn.Module) -> None:
        self.caduceus = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                weight = torch.tensor(loss_weights, dtype=logits.dtype, device=logits.device)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    weight=weight,
                    ignore_index=self.config.pad_token_id,
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.config.pad_token_id,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel):
    """Caduceus pooled-backbone sequence classifier with optional RC-conjoin."""

    def __init__(
        self,
        config: CaduceusConfig,
        pooling_strategy: str = "mean",
        conjoin_train: bool = False,
        conjoin_eval: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(
                f"Pooling strategy '{pooling_strategy}' is not implemented. "
                "Choose from: 'mean', 'max', 'first', 'last'."
            )
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)
        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval
        self.post_init()
        self.init_scorer()

    def init_scorer(self, initializer_range: float = 0.02) -> None:
        initializer_range = (
            self.config.initializer_cfg.get("initializer_range", initializer_range)
            if self.config.initializer_cfg is not None
            else initializer_range
        )
        self.score.weight.data.normal_(std=initializer_range)

    def get_input_embeddings(self) -> nn.Module:
        """Return the word embedding module."""
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        if self.config.rcps:
            raise NotImplementedError(
                "Setting input embeddings is not supported in RCPS mode."
            )
        self.caduceus.backbone.embeddings.word_embeddings = value

    def pool_hidden_states(
        self, hidden_states: torch.Tensor, sequence_length_dim: int = 1
    ) -> torch.Tensor:
        if self.pooling_strategy == "mean":
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":
            return hidden_states.movedim(sequence_length_dim, 0)
        if self.pooling_strategy == "first":
            return hidden_states.movedim(sequence_length_dim, 0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.rcps:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., : self.config.d_model],
                    torch.flip(
                        transformer_outputs[0][..., self.config.d_model :], dims=[1, 2]
                    ),
                ],
                dim=-1,
            )
        elif self.conjoin_train or (self.conjoin_eval and not self.training):
            if input_ids is None:
                raise ValueError("`input_ids` must be provided for conjoining.")
            if input_ids.ndim != 3:
                raise ValueError(
                    "`input_ids` must be a 3-D tensor of shape [B, L, 2] "
                    "(forward and RC strands) when conjoin mode is active."
                )
            transformer_outputs = self.caduceus(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.caduceus(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [transformer_outputs[0], transformer_outputs_rc[0]], dim=-1
            )
        else:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        if hidden_states.ndim == 4:  # stacked fwd/rc reps: [B, L, D, 2]
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = (
                    F.mse_loss(logits.squeeze(), labels.squeeze())
                    if self.num_labels == 1
                    else F.mse_loss(logits, labels)
                )
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
