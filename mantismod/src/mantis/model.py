from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvEmbedding(nn.Module):
    """
    Multi-scale convolutional embedding that captures temporal patterns at
    multiple resolutions simultaneously.

    Four parallel ``Conv1d`` branches with kernel sizes 3, 7, 15, and 31
    extract short-, medium-, long-, and very-long-range features
    respectively. Each branch is independently normalised and activated,
    and the four feature maps are concatenated then projected back to
    *output_dim* via a final linear layer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            input_dim: Number of input channels (1 for no-covariate
                models, 2 for covariate models).
            output_dim: Width of the output embedding. Must be divisible
                by 4 because each branch produces ``output_dim // 4``
                channels.
            dropout: Dropout probability applied after each branch's
                activation.
        """
        super().__init__()

        # Parallel convolutions at four temporal scales
        self.conv_short = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3,  padding=1)
        self.conv_med   = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7,  padding=3)
        self.conv_long  = nn.Conv1d(input_dim, output_dim // 4, kernel_size=15, padding=7)
        self.conv_vlong = nn.Conv1d(input_dim, output_dim // 4, kernel_size=31, padding=15)

        # Per-branch layer normalisation
        self.norm_short = nn.LayerNorm(output_dim // 4)
        self.norm_med   = nn.LayerNorm(output_dim // 4)
        self.norm_long  = nn.LayerNorm(output_dim // 4)
        self.norm_vlong = nn.LayerNorm(output_dim // 4)

        self.activation = nn.SiLU()
        self.dropout    = nn.Dropout(dropout)

        # Final projection combining all four scales
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a multivariate time series using parallel multi-scale convolutions.

        Args:
            x: Input tensor of shape ``[batch_size, seq_len, input_dim]``.

        Returns:
            Embedded tensor of shape ``[batch_size, seq_len, output_dim]``.
        """
        # Conv1d expects [B, C, T]; transpose then transpose back
        x_conv = x.transpose(1, 2)

        x_short = self.conv_short(x_conv).transpose(1, 2)
        x_med   = self.conv_med(x_conv).transpose(1, 2)
        x_long  = self.conv_long(x_conv).transpose(1, 2)
        x_vlong = self.conv_vlong(x_conv).transpose(1, 2)

        x_short = self.dropout(self.activation(self.norm_short(x_short)))
        x_med   = self.dropout(self.activation(self.norm_med(x_med)))
        x_long  = self.dropout(self.activation(self.norm_long(x_long)))
        x_vlong = self.dropout(self.activation(self.norm_vlong(x_vlong)))

        x_combined = torch.cat([x_short, x_med, x_long, x_vlong], dim=2)
        x_out = self.final_norm(self.final_proj(x_combined))
        return x_out


class TemporalPatternAttention(nn.Module):
    """
    Multi-head self-attention augmented with a learnable relative
    positional bias.

    Standard scaled dot-product attention scores are additively adjusted
    by a bias table indexed by the relative distance between every pair
    of sequence positions. This encourages the model to attend
    preferentially to temporally nearby or periodically related positions.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        max_len: int = 2000,
    ) -> None:
        """
        Args:
            hidden_dim: Total dimension of query, key, and value
                projections. Must be divisible by *n_heads*.
            n_heads: Number of parallel attention heads.
            dropout: Dropout probability applied to attention weights
                after the softmax.
            max_len: Maximum sequence length supported by the relative
                positional bias table.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable relative positional bias: one scalar per head per offset
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
        positions = torch.arange(max_len).unsqueeze(1) - torch.arange(max_len).unsqueeze(0)
        positions = positions + max_len - 1  # shift to [0, 2*max_len-2]
        self.register_buffer("positions", positions)

        self.dropout = nn.Dropout(dropout)
        self.scale   = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply relative-position-biased multi-head self-attention.

        Args:
            x: Input tensor of shape
                ``[batch_size, seq_len, hidden_dim]``.
            mask: Optional boolean tensor of shape
                ``[batch_size, seq_len]`` where ``True`` marks valid
                (non-padding) positions. Positions where ``mask`` is
                ``False`` are excluded from attention by being filled
                with ``-inf`` before the softmax.

        Returns:
            Output tensor of shape ``[batch_size, seq_len, hidden_dim]``.
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Relative positional bias: [n_heads, seq_len, seq_len]
        rel_pos = self.rel_pos_bias[self.positions[:seq_len, :seq_len]]  # [T, T, H]
        attn_scores = attn_scores + rel_pos.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            invalid = (~mask).unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, seq_len, -1)
            attn_scores = attn_scores.masked_fill(invalid, float("-inf"))

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_weights, v)                            # [B, H, T, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)


class CNNTransformerBlock(nn.Module):
    """
    Hybrid encoder block that combines depthwise convolution for local
    pattern extraction with :class:`TemporalPatternAttention` for global
    context, followed by a position-wise feed-forward network.

    Uses a post-normalisation (post-norm) residual architecture throughout.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        """
        Args:
            hidden_dim: Feature dimension used throughout the block.
            ffn_dim: Hidden dimension of the two-layer feed-forward
                network (typically 2–4× *hidden_dim*).
            n_heads: Number of attention heads in
                :class:`TemporalPatternAttention`.
            dropout: Dropout probability applied after each sub-layer
                before the residual addition.
            layer_norm_eps: Epsilon value for all ``LayerNorm`` layers
                in this block.
        """
        super().__init__()

        # Depthwise-separable convolution for local context
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.pattern_attn = TemporalPatternAttention(hidden_dim, n_heads, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass a sequence through one CNN-Transformer block.

        Args:
            x: Input tensor of shape
                ``[batch_size, seq_len, hidden_dim]``.
            mask: Optional boolean mask of shape
                ``[batch_size, seq_len]`` (``True`` = valid position).
                Forwarded unchanged to :class:`TemporalPatternAttention`.

        Returns:
            Output tensor of the same shape as *x*:
            ``[batch_size, seq_len, hidden_dim]``.
        """
        # Local depthwise convolution sub-layer (post-norm)
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + self.dropout(x_conv))

        # Global attention sub-layer (post-norm)
        x = self.norm2(x + self.dropout(self.pattern_attn(x, mask)))

        # Feed-forward sub-layer (post-norm)
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class EpidemicPatternMemory(nn.Module):
    """
    Differentiable external memory bank of learnable disease-pattern
    prototypes.

    The module maintains a fixed-size bank of ``num_patterns`` prototype
    vectors. At each sequence position the module computes a soft
    attention distribution over all prototypes (conditioned on the local
    hidden state) and adds the retrieved weighted combination back to the
    hidden state via a residual connection. This allows the model to
    incorporate generalised knowledge about recurring epidemic shapes
    (seasonal surges, intervention effects, reporting artefacts, etc.)
    independently of the input sequence.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_patterns: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        """
        Args:
            hidden_dim: Dimensionality of both the hidden states and each
                stored prototype vector.
            num_patterns: Number of learnable prototype vectors in the
                memory bank.
            dropout: Dropout probability applied to the retrieved pattern
                vectors before the residual addition.
            layer_norm_eps: Epsilon value for the two ``LayerNorm``
                layers.
        """
        super().__init__()

        # Learnable prototype bank
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        # Content-based pattern matching: hidden state → attention logits
        self.pattern_matcher = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns),
        )

        self.pattern_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1   = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2   = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Augment encoder hidden states with retrieved pattern prototypes.

        Args:
            x: Encoder output tensor of shape
                ``[batch_size, seq_len, hidden_dim]``.
            mask: Optional boolean tensor of shape
                ``[batch_size, seq_len]`` (``True`` = valid position).
                Attention logits at masked-out positions are zeroed
                before the softmax so they do not contribute to the
                retrieved pattern.

        Returns:
            Pattern-augmented tensor of the same shape as *x*:
            ``[batch_size, seq_len, hidden_dim]``.
        """
        x_norm = self.norm1(x)

        # Compute attention weights over the pattern bank
        pattern_weights = self.pattern_matcher(x_norm)  # [B, T, num_patterns]
        if mask is not None:
            pattern_weights = pattern_weights * mask.unsqueeze(-1).float()
        pattern_weights = F.softmax(pattern_weights, dim=-1)

        # Retrieve and project weighted prototype combination
        retrieved = self.dropout(
            self.pattern_proj(torch.matmul(pattern_weights, self.pattern_bank))
        )

        return self.norm2(x + retrieved)


class MultiTimeSeriesForecaster(nn.Module):
    """
    Foundation model for probabilistic disease-outbreak forecasting.

    Implements a hybrid CNN-Transformer encoder followed by an
    autoregressive GRU decoder that emits multi-quantile predictions.
    Key design choices:

    * **Multi-scale convolutional embedding** to capture patterns across
      different temporal granularities.
    * **Relative-position-biased self-attention** in each encoder block.
    * **Epidemic pattern memory bank** appended after the encoder to
      recall recurring outbreak shapes.
    * **Two-layer GRU decoder** with cross-attention to the encoder memory
      for autoregressive quantile prediction.
    * Supports both single-channel (no-covariate) and two-channel
      (with-covariate) inputs via the *values_input_dim* argument.
    """

    def __init__(
        self,
        input_window: int = 112,
        forecast_horizon: int = 4,
        hidden_dim: int = 512,
        ffn_dim: int = 768,
        n_layers: int = 8,
        n_heads: int = 8,
        n_quantiles: int = 9,
        disease_embed_dim: int = 64,
        pop_embed_dim: int = 64,
        binary_feat_dim: int = 32,
        teacher_forcing_ratio: float = 0.1,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        values_input_dim: int = 2,
    ) -> None:
        """
        Args:
            input_window: Expected length of the input sequence in weeks.
                Defaults to ``112`` (≈ 2 years of weekly data).
            forecast_horizon: Number of future weeks to predict.
                Must match the loaded checkpoint; typically ``4`` or ``8``.
            hidden_dim: Core feature dimension used throughout the encoder
                and decoder.
            ffn_dim: Hidden dimension of the feed-forward sublayer inside
                each :class:`CNNTransformerBlock`.
            n_layers: Number of stacked :class:`CNNTransformerBlock`
                encoder layers.
            n_heads: Number of attention heads in each encoder block and
                in the decoder cross-attention layer.
            n_quantiles: Number of output quantiles per forecast step.
                The model produces one independent projection head per
                quantile.
            disease_embed_dim: Output dimension of the disease-type
                embedding table (3 disease categories).
            pop_embed_dim: Output dimension of the population MLP.
            binary_feat_dim: Reserved dimension for binary feature
                embeddings (not currently used in forward pass but
                retained for checkpoint compatibility).
            teacher_forcing_ratio: During training, the probability of
                feeding the ground-truth value rather than the model's
                own median prediction as the next decoder input.
            dropout: Dropout probability applied uniformly in all
                sub-layers.
            layer_norm_eps: Epsilon used in all ``LayerNorm`` layers.
            values_input_dim: Number of input channels for the time-series
                embedding (``1`` = target only, ``2`` = target +
                covariate). Must match the checkpoint.
        """
        super().__init__()
        self.input_window         = input_window
        self.forecast_horizon     = forecast_horizon
        self.hidden_dim           = hidden_dim
        self.n_layers             = n_layers
        self.n_heads              = n_heads
        self.n_quantiles          = n_quantiles
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # ===== FEATURE EMBEDDINGS =========================================
        self.values_embedding = MultiScaleConvEmbedding(
            input_dim=values_input_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self.disease_embedding = nn.Embedding(3, disease_embed_dim)
        self.disease_norm      = nn.LayerNorm(disease_embed_dim, eps=layer_norm_eps)

        self.population_mlp = nn.Sequential(
            nn.Linear(1, pop_embed_dim),
            nn.LayerNorm(pop_embed_dim, eps=layer_norm_eps),
            nn.GELU(),
        )

        # Temporal embeddings derived from absolute day indices
        self.day_of_week_embed  = nn.Embedding(7,   128)
        self.day_norm           = nn.LayerNorm(128, eps=layer_norm_eps)
        self.month_embed        = nn.Embedding(12,  128)
        self.month_norm         = nn.LayerNorm(128, eps=layer_norm_eps)
        self.day_of_year_embed  = nn.Embedding(366, 128)
        self.day_of_year_norm   = nn.LayerNorm(128, eps=layer_norm_eps)

        self.target_type_embedding = nn.Embedding(3, 128)
        self.target_type_norm      = nn.LayerNorm(128, eps=layer_norm_eps)

        # Combined feature projection
        input_feat_dim = hidden_dim + disease_embed_dim + pop_embed_dim + 128 + 128 + 128 + 128
        self.feature_projection = nn.Linear(input_feat_dim, hidden_dim)
        self.feature_norm       = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # ===== ENCODER ====================================================
        self.encoder_blocks = nn.ModuleList([
            CNNTransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                n_heads=n_heads,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(n_layers)
        ])

        # ===== PATTERN MEMORY =============================================
        self.pattern_memory = EpidemicPatternMemory(
            hidden_dim=hidden_dim,
            num_patterns=256,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )

        # ===== DECODER ====================================================
        self.decoder_init_proj = nn.Linear(hidden_dim, hidden_dim)

        self.decoder_input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4, eps=layer_norm_eps),
            nn.GELU(),
        )

        self.decoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder_gru = nn.GRU(
            input_size=hidden_dim + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # One independent projection head per output quantile
        self.quantile_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2, eps=layer_norm_eps),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(n_quantiles)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise model parameters with conservative values.

        Weight matrices receive Xavier uniform initialisation with a
        reduced gain (``0.01``) to stabilise early training. Biases are
        zeroed. Embedding weights and pattern-bank entries are sampled
        from narrow normal distributions.
        """
        for name, p in self.named_parameters():
            if "weight" in name and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p, gain=0.01)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "embedding" in name:
                nn.init.normal_(p, mean=0.0, std=0.01)
            elif "pattern_bank" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        disease_type: torch.Tensor,
        target_type: torch.Tensor,
        population: torch.Tensor,
        day_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode, retrieve patterns, decode autoregressively.

        Args:
            values: Normalised input time series of shape
                ``[batch_size, seq_len, C]`` where ``C`` is 1 (no
                covariate) or 2 (with covariate).
            disease_type: Disease-category indices of shape
                ``[batch_size]`` (``LongTensor``). Currently a placeholder
                (always ``0``).
            target_type: Target-signal indices of shape ``[batch_size]``
                (``LongTensor``): ``0`` = cases, ``1`` = hospitalisations,
                ``2`` = deaths.
            population: Log-transformed (and optionally z-scored)
                population values of shape ``[batch_size]``
                (``FloatTensor``).
            day_indices: Absolute day indices of shape
                ``[batch_size, seq_len]`` (``LongTensor``), used to
                derive day-of-week, month, and day-of-year embeddings.
            valid_mask: Optional boolean tensor of shape
                ``[batch_size, seq_len]`` where ``True`` marks valid
                (non-padding) positions. When ``None`` all positions are
                treated as valid.
            target_values: Optional ground-truth targets of shape
                ``[batch_size, forecast_horizon]`` used for teacher
                forcing during training. Ignored at inference time.

        Returns:
            Quantile predictions of shape
            ``[batch_size, forecast_horizon, n_quantiles]``.
        """
        batch_size, seq_len, _ = values.shape
        device: torch.device = values.device

        # ===== INPUT PROCESSING ==========================================
        value_features = self.values_embedding(values)

        disease_emb = self.disease_norm(self.disease_embedding(disease_type))  # [B, D_d]
        pop_emb     = self.population_mlp(population.unsqueeze(-1))            # [B, D_p]

        # Temporal features derived from absolute day indices
        dow_emb = self.day_norm(self.day_of_week_embed((day_indices % 7).long()))
        month_emb = self.month_norm(self.month_embed(((day_indices // 30) % 12).long()))
        doy_emb = self.day_of_year_norm(self.day_of_year_embed((day_indices % 366).long()))

        target_type_emb = self.target_type_norm(self.target_type_embedding(target_type))

        # Broadcast static embeddings over the time dimension
        disease_emb_exp     = disease_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        pop_emb_exp         = pop_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        target_type_emb_exp = target_type_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        combined_features = torch.cat(
            [value_features, disease_emb_exp, pop_emb_exp,
             target_type_emb_exp, dow_emb, month_emb, doy_emb],
            dim=-1,
        )

        encoder_input = self.feature_norm(self.feature_projection(combined_features))

        # ===== ENCODER ===================================================
        encoder_output: torch.Tensor = encoder_input
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output, valid_mask)

        # ===== PATTERN MEMORY ============================================
        encoder_output = self.pattern_memory(encoder_output, valid_mask)

        # ===== AUTOREGRESSIVE DECODER ====================================
        # Initialise decoder hidden state from masked mean pooling
        if valid_mask is not None:
            mask_exp   = valid_mask.unsqueeze(-1).float()
            pooled_state = (encoder_output * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-10)
        else:
            pooled_state = encoder_output.mean(dim=1)  # [B, hidden_dim]

        # Two-layer GRU initial hidden state: [2, B, hidden_dim]
        dec_hidden: torch.Tensor = (
            self.decoder_init_proj(pooled_state).unsqueeze(0).repeat(2, 1, 1)
        )

        # Seed decoder with the last observed (normalised) target value
        decoder_input: torch.Tensor = (
            values[:, -1, 0].unsqueeze(-1)
            if seq_len > 0
            else torch.zeros(batch_size, 1, device=device)
        )

        use_teacher_forcing: bool = (
            self.training
            and target_values is not None
            and torch.rand(1).item() < self.teacher_forcing_ratio
        )

        all_quantile_preds: list[torch.Tensor] = []

        for t in range(self.forecast_horizon):
            dec_input_emb = self.decoder_input_proj(decoder_input.unsqueeze(-1))  # [B, D//4]
            if dec_input_emb.dim() == 2:
                dec_input_emb = dec_input_emb.unsqueeze(1)                        # [B, 1, D//4]

            query = dec_hidden[-1:].transpose(0, 1)                               # [B, 1, D]
            attn_output, _ = self.decoder_attn(
                query, encoder_output, encoder_output,
                key_padding_mask=None if valid_mask is None else ~valid_mask,
            )

            gru_input = torch.cat([attn_output, dec_input_emb], dim=-1)           # [B, 1, D+D//4]
            _, dec_hidden = self.decoder_gru(gru_input, dec_hidden)

            dec_output = dec_hidden[-1, :, :]                                     # [B, D]
            step_pred = torch.cat(
                [q_proj(dec_output) for q_proj in self.quantile_projections], dim=1
            )                                                                      # [B, n_quantiles]
            all_quantile_preds.append(step_pred.unsqueeze(1))                     # [B, 1, Q]

            median_idx = self.n_quantiles // 2
            if use_teacher_forcing and t < target_values.size(1):
                decoder_input = target_values[:, t].unsqueeze(-1)
            else:
                decoder_input = step_pred[:, median_idx].unsqueeze(-1)

        predictions = torch.cat(all_quantile_preds, dim=1)  # [B, H, Q]
        return predictions

    def predict(
        self,
        values: torch.Tensor,
        disease_type: torch.Tensor,
        target_type: torch.Tensor,
        population: torch.Tensor,
        day_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience inference wrapper that disables teacher forcing and
        gradient computation.

        Temporarily sets the model to evaluation mode if it is currently
        in training mode, runs a forward pass, then restores the original
        training state.

        Args:
            values: Normalised input tensor of shape
                ``[batch_size, seq_len, C]``.
            disease_type: Disease-category index tensor of shape
                ``[batch_size]`` (``LongTensor``).
            target_type: Target-signal index tensor of shape
                ``[batch_size]`` (``LongTensor``).
            population: Population tensor of shape ``[batch_size]``
                (``FloatTensor``).
            day_indices: Absolute day-index tensor of shape
                ``[batch_size, seq_len]`` (``LongTensor``).
            valid_mask: Optional boolean validity mask of shape
                ``[batch_size, seq_len]``.

        Returns:
            Quantile predictions of shape
            ``[batch_size, forecast_horizon, n_quantiles]`` with no
            gradient attached.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(
                values,
                disease_type,
                target_type,
                population,
                day_indices,
                valid_mask=valid_mask,
                target_values=None,
            )
        if was_training:
            self.train()
        return preds
