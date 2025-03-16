from functools import partial
import flax.linen as nn
import jax.numpy as jnp

import einops


import numpy as np

from flax.linen.initializers import constant, orthogonal


class GRUGating(nn.Module):

    dim: int
    scale_residual: bool = False

    def setup(self):
        super().__init__()
        self.gru = nn.GRUCell(self.dim, self.dim)
        self.residual_scale = nn.Parameter(
            jnp.ones(self.dim)) if self.scale_residual else None

    def __call__(self, x, residual):
        if self.residual_scale is not None:
            residual = residual * self.residual_scale

        gated_output = self.gru(
            einops.rearrange(x, 'b n d -> (b n) d'),
            einops.rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


class GatedEncoderBlock(nn.Module):
    # Input dimension is needed here since it is equal to the output dimension (residual connection)
    hidden_dim: int
    num_heads: int
    dim_feedforward: int
    init_scale: float
    use_fast_attention: bool
    dropout_prob: float = 0.

    def setup(self):
        # Attention layer
        if self.use_fast_attention:
            from fast_attention import make_fast_generalized_attention
            raw_attention_fn = make_fast_generalized_attention(
                self.hidden_dim // self.num_heads,
                renormalize_attention=True,
                nb_features=self.hidden_dim,
                unidirectional=False
            )
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                attention_fn=raw_attention_fn,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        else:
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward, kernel_init=nn.initializers.xavier_uniform(
            ), bias_init=constant(0.0)),
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(
            ), bias_init=constant(0.0))
        ]
        # Layers to apply in between the main layers
        self.gate1 = GRUGating(dim=self.hidden_dim)
        self.norm1 = nn.LayerNorm()
        self.gate2 = GRUGating(dim=self.hidden_dim)
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        # masking is not compatible with fast self attention
        x_norm1 = self.norm1(x)
        if mask is not None and not self.use_fast_attention:
            mask = jnp.repeat(nn.make_attention_mask(
                mask, mask), self.num_heads, axis=-3)
        attended = self.self_attn(
            inputs_q=x_norm1, inputs_kv=x_norm1, mask=mask, deterministic=deterministic)

        # GRU gate
        x = self.gate1(attended, x_norm1)
        x = self.dropout(x, deterministic=deterministic)

        x_res = x

        # MLP part
        x = self.norm2(x)
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        # GRU Gate
        x = self.gate2(x, x_res)
        x = self.dropout(x, deterministic=deterministic)
        return x


class EncoderBlock(nn.Module):
    # Input dimension is needed here since it is equal to the output dimension (residual connection)
    hidden_dim: int
    num_heads: int
    dim_feedforward: int
    init_scale: float
    use_fast_attention: bool
    dropout_prob: float = 0.

    def setup(self):
        # Attention layer
        if self.use_fast_attention:
            from fast_attention import make_fast_generalized_attention
            raw_attention_fn = make_fast_generalized_attention(
                self.hidden_dim // self.num_heads,
                renormalize_attention=True,
                nb_features=self.hidden_dim,
                unidirectional=False
            )
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                attention_fn=raw_attention_fn,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        else:
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward, kernel_init=nn.initializers.xavier_uniform(
            ), bias_init=constant(0.0)),
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(
            ), bias_init=constant(0.0))
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        # masking is not compatible with fast self attention
        if mask is not None and not self.use_fast_attention:
            mask = jnp.repeat(nn.make_attention_mask(
                mask, mask), self.num_heads, axis=-3)
        attended = self.self_attn(
            inputs_q=x, inputs_kv=x, mask=mask, deterministic=deterministic)

        x = self.norm1(attended + x)
        x = x + self.dropout(x, deterministic=deterministic)

        # MLP part
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        x = self.norm2(feedforward+x)
        x = x + self.dropout(x, deterministic=deterministic)

        return x


class Embedder(nn.Module):
    hidden_dim: int
    init_scale: float
    scale_inputs: bool = True
    activation: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        if self.scale_inputs:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(
            self.init_scale), bias_init=constant(0.0))(x)
        if self.activation:
            x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class ScannedTransformer(nn.Module):

    hidden_dim: int
    init_scale: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float = 0
    deterministic: bool = True
    return_embeddings: bool = False
    use_fast_attention: bool = False
    gated: bool = True

    def setup(self):
        self.encoders = [
            GatedEncoderBlock(
                self.hidden_dim,
                self.transf_num_heads,
                self.transf_dim_feedforward,
                self.init_scale,
                self.use_fast_attention,
                self.transf_dropout_prob,
            ) if self.gated else EncoderBlock(
                self.hidden_dim,
                self.transf_num_heads,
                self.transf_dim_feedforward,
                self.init_scale,
                self.use_fast_attention,
                self.transf_dropout_prob,
            ) for _ in range(self.transf_num_layers)
        ]

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        hs = carry
        embeddings, mask, done = x

        hs = jnp.where(
            done[:, np.newaxis, np.newaxis],
            self.initialize_carry(self.hidden_dim, *done.shape, 1),
            hs
        )
        embeddings = jnp.concatenate((
            hs,
            embeddings,
        ), axis=-2)
        for layer in self.encoders:
            embeddings = layer(embeddings, mask=mask,
                               deterministic=self.deterministic)
        hs = embeddings[..., 0:1, :]

        # as y return the entire embeddings if required (i.e. transformer mixer), otherwise only agents' hs embeddings
        if self.return_embeddings:
            return hs, embeddings
        else:
            return hs, hs

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return jnp.zeros((*batch_size, hidden_size))
