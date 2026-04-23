"""CaduceusConfig: HF PretrainedConfig subclass controlling architecture
(SSM dim, depth, bidirectionality, RCPS), frame-positional encoding, and
trainer-side hidden-state cache settings."""

from typing import Dict, Optional

from transformers import PretrainedConfig


class CaduceusConfig(PretrainedConfig):
    """HF config for the Caduceus bidirectional Mamba backbone."""

    model_type = "caduceus"

    def __init__(
        self,
        d_model: int = 192,
        n_layer: int = 8,
        vocab_size: int = 6,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = False,
        pad_vocab_size_multiple: int = 8,
        norm_epsilon: float = 1e-5,
        initializer_cfg: Optional[dict] = None,
        # BiMamba settings
        bidirectional: bool = True,
        bidirectional_strategy: str = "add",  # "add" | "gated" | "concat_linear"
        bidirectional_weight_tie: bool = True,
        # RCPS (optional)
        rcps: bool = False,
        complement_map: Optional[Dict[int, int]] = None,
        # Output settings
        use_hidden_states: bool = True,
        # Frame-positional encoding
        use_frame_positional_encoding: bool = True,
        frame_pos_mode: str = "off",  # "off" | "local" | "global"
        # Hidden state caching (used by trainer)
        hidden_state_size: int = 192,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg or {}

        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg

        # BiMamba settings
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie

        # RCPS and complement mapping
        self.rcps = rcps
        self.complement_map = complement_map or {
            0: 1,
            1: 0,  # A <-> T
            2: 3,
            3: 2,  # C <-> G
            4: 4,  # N -> N
            5: 5,  # PAD -> PAD
        }

        # Output settings
        self.use_hidden_states = use_hidden_states

        # Frame positional encoding policy
        self.use_frame_positional_encoding = bool(use_frame_positional_encoding)
        if frame_pos_mode not in ("off", "local", "global"):
            raise ValueError(
                f"frame_pos_mode must be one of 'off', 'local', 'global', "
                f"got {frame_pos_mode!r}"
            )
        self.frame_pos_mode = frame_pos_mode

        # Hidden-state cache size (used by trainer when use_hidden_cache=True)
        self.hidden_state_size = hidden_state_size

        self._validate_complement_map()

    def _validate_complement_map(self) -> None:
        required_pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 4), (5, 5)}
        existing_pairs = set(self.complement_map.items())
        if not required_pairs.issubset(existing_pairs):
            missing = required_pairs - existing_pairs
            raise ValueError(f"Missing required complement pairs: {missing}")
