"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple, Sequence

import einops
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from tensorflow_probability.substrates import jax as tfp

from minimax.models import common
from minimax.models import s5
from minimax.models import transformer
from minimax.models.registration import register
from minimax.models.moe import SoftMoE

from flax.linen.initializers import constant, orthogonal


class BasicModel(nn.Module):
    """Split Actor-Critic Architecture for PPO."""
    output_dim: int = 6
    n_hidden_layers: int = 1
    hidden_dim: int = 32
    n_conv_layers: int = 1
    n_conv_filters: int = 16
    conv_encoder: bool = True
    conv_kernel_size: int = 3
    n_scalar_embeddings: int = 4
    max_scalar: int = 4
    scalar_embed_dim: int = 5
    recurrent_arch: str = None
    recurrent_hidden_dim: int = 256
    base_activation: str = 'relu'
    head_activation: str = 'tanh'

    s5_n_blocks: int = 2
    s5_n_layers: int = 4
    s5_layernorm_pos: str = None
    s5_activation: str = "half_glu1"

    transf_init_scale: float = 0.1
    transf_num_layers: int = 2
    transf_num_heads: int = 4
    transf_dropout_prob: float = 0.0
    transf_deterministic: bool = True
    transf_return_embeddings: bool = False
    transf_use_fast_attention: bool = False
    transf_gated: bool = True

    is_soft_moe: bool = False
    soft_moe_num_experts: int = 4
    soft_moe_num_slots: int = 32

    value_ensemble_size: int = 1

    def setup(self):

        if self.conv_encoder:
            conv_list = []
            for i, feat in enumerate([32, 64, 32]):
                # padding = "SAME" if i < self.n_conv_layers - 2 else "VALID"
                conv_list.append(
                    nn.Conv(
                        features=feat,
                        kernel_size=[self.conv_kernel_size,]*2,
                        strides=1,
                        kernel_init=common.init_orth(
                            scale=common.calc_gain(self.base_activation)
                        ),
                        bias_init=common.default_bias_init(),
                        padding=((1, 1), (1, 1)),  # padding,  # 'SAME',
                        name=f'cnn_{i}'
                    )
                )
                conv_list.append(
                    common.get_activation(self.base_activation)
                )

            self.conv = nn.Sequential(conv_list)
            self.after_conv = common.make_fc_layers(
                n_layers=self.n_hidden_layers,
                hidden_dim=self.hidden_dim,
                activation=common.get_activation(self.base_activation),
                kernel_init=common.init_orth(
                    scale=common.calc_gain(self.base_activation)
                ),
                bias_init=common.default_bias_init(),
                use_layernorm=True,
            )
            self.linear_encoder = None
        else:
            self.conv = None
            self.linear_encoder = nn.Sequential([
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=common.init_orth(
                        common.calc_gain('linear')
                    ),
                    name=f'fc_linear'
                ),
                common.get_activation(self.base_activation),
                nn.LayerNorm(name='ln_linear'),
            ])

        if self.is_soft_moe:
            self.moe = SoftMoE(
                in_features=self.n_conv_filters,
                out_features=self.hidden_dim,
                num_experts=self.soft_moe_num_experts,  # 4,
                slots_per_expert=self.soft_moe_num_slots,  # 32,
            )

        if self.n_scalar_embeddings > 0:
            self.fc_scalar = nn.Embed(
                num_embeddings=self.n_scalar_embeddings,
                features=self.scalar_embed_dim,
                embedding_init=common.init_orth(
                    common.calc_gain('linear')
                ),
                name=f'fc_scalar'
            )
        elif self.scalar_embed_dim > 0:
            self.fc_scalar = nn.Dense(
                self.scalar_embed_dim,
                kernel_init=common.init_orth(
                    common.calc_gain('linear')
                ),
                name=f'fc_scalar'
            )
        else:
            self.fc_scalar = None

        if self.recurrent_arch is not None:
            if self.recurrent_arch == 's5':
                self.embed_pre_s5 = nn.Sequential([
                    nn.Dense(
                        self.recurrent_hidden_dim,
                        kernel_init=common.init_orth(
                            common.calc_gain('linear')
                        ),
                        name=f'fc_pre_s5'
                    )
                ])
                self.rnn = s5.make_s5_encoder_stack(
                    input_dim=self.recurrent_hidden_dim,
                    ssm_state_dim=self.recurrent_hidden_dim,
                    n_blocks=self.s5_n_blocks,
                    n_layers=self.s5_n_layers,
                    activation=self.s5_activation,
                    layernorm_pos=self.s5_layernorm_pos
                )
            elif self.recurrent_arch == 'transformer':
                self.rnn = transformer.ScannedTransformer(
                    hidden_dim=self.recurrent_hidden_dim,
                    init_scale=self.transf_init_scale,
                    transf_num_layers=self.transf_num_layers,
                    transf_num_heads=self.transf_num_heads,
                    transf_dim_feedforward=self.recurrent_hidden_dim,
                    transf_dropout_prob=self.transf_transf_dropout_prob,
                    deterministic=self.transf_deterministic,
                    return_embeddings=self.transf_return_embeddings,
                    use_fast_attention=self.transf_use_fast_attention,
                    gated=self.transf_gated,
                )
            else:
                self.rnn = common.ScannedRNN(
                    recurrent_arch=self.recurrent_arch,
                    recurrent_hidden_dim=self.recurrent_hidden_dim,
                    kernel_init=common.init_orth(),
                    recurrent_kernel_init=common.init_orth()
                )
        else:
            self.rnn = None

        self.pi_head = nn.Sequential([
            # common.make_fc_layers(
            #     'fc_pi',
            #     n_layers=self.n_hidden_layers,
            #     hidden_dim=self.hidden_dim,
            #     activation=common.get_activation(self.head_activation),
            #     kernel_init=common.init_orth(
            #         common.calc_gain(self.head_activation)
            #     )
            # ),
            nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.constant(0.01),
                name=f'fc_pi_final'
            )
        ])

        value_head_kwargs = dict(
            n_hidden_layers=0,
            hidden_dim=self.hidden_dim,
            activation=nn.tanh,
            kernel_init=common.init_orth(
                common.calc_gain('tanh')
            ),
            last_layer_kernel_init=common.init_orth(
                common.calc_gain('linear')
            )
        )

        if self.value_ensemble_size > 1:
            self.v_head = common.EnsembleValueHead(
                n=self.value_ensemble_size, **value_head_kwargs)
        else:
            self.v_head = common.ValueHead(**value_head_kwargs)

    def __call__(self, x, carry=None):
        raise NotImplementedError

    def initialize_carry(
            self,
            rng: chex.PRNGKey,
            batch_dims: Tuple[int] = ()) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initialize hidden state of LSTM."""
        if self.recurrent_arch is not None:
            if self.recurrent_arch == 's5':
                return s5.S5EncoderStack.initialize_carry(  # Since conj_sym=True
                    rng, batch_dims, self.recurrent_hidden_dim//2, self.s5_n_layers
                )
            elif self.recurrent_arch == 'transformer':
                return transformer.ScannedTransformer.initialize_carry(
                    self.recurrent_hidden_dim, batch_dims)
            else:
                return common.ScannedRNN.initialize_carry(
                    rng, batch_dims, self.recurrent_hidden_dim, self.recurrent_arch)
        else:
            raise ValueError('Model is not recurrent.')

    @property
    def is_recurrent(self):
        return self.recurrent_arch is not None


class ACStudentActorModel(BasicModel):
    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x

        if self.rnn is not None:
            batch_dims = img.shape[:-3]
            x = self.conv(img)
        else:
            batch_dims = img.shape[:-3]
            x = self.conv(img)

        if self.is_soft_moe:
            initial_shape = x.shape
            if len(initial_shape) == 5:
                a, n, h, w, f = x.shape
                x = einops.rearrange(x, "a n ... -> (a n) ...", a=a, n=n)

            x = einops.rearrange(x, "... w h c -> ... (w h) c")

            x = self.moe(x)

            if len(initial_shape) == 5:
                x = einops.rearrange(x, "(a n) ... -> a n ...", a=a, n=n)

        x = x.reshape(*batch_dims, -1)
        x = self.after_conv(x)

        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            elif self.recurrent_arch == 'transformer':
                x = self.rnn(carry, (x, mask, reset))
            else:
                carry, x = self.rnn(carry, (x, reset))

        logits = self.pi_head(x)
        return logits, carry


class ACStudentActorModelMlp(BasicModel):

    conv_encoder: bool = False

    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x

        if self.rnn is not None:
            batch_dims = img.shape[:-1]
            x = self.linear_encoder(img)
            x = x.reshape(*batch_dims, -1)
        else:
            batch_dims = img.shape[:-1]
            x = self.linear_encoder(img)
            x = x.reshape(*batch_dims, -1)

        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            elif self.recurrent_arch == 'transformer':
                x = self.rnn(carry, (x, mask, reset))
            else:
                carry, x = self.rnn(carry, (x, reset))

        logits = self.pi_head(x)
        return logits, carry


class ACStudentCriticModel(BasicModel):

    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x

        if self.rnn is not None:
            batch_dims = img.shape[:-3]
            x = self.conv(img)
        else:
            batch_dims = img.shape[:-3]
            x = self.conv(img)

        if self.is_soft_moe:
            initial_shape = x.shape
            if len(initial_shape) == 5:
                a, n, h, w, f = x.shape
                x = einops.rearrange(x, "a n ... -> (a n) ...", a=a, n=n)

            x = einops.rearrange(x, "... w h c -> ... (w h) c")

            x = self.moe(x)

            if len(initial_shape) == 5:
                x = einops.rearrange(x, "(a n) ... -> a n ...", a=a, n=n)

        x = x.reshape(*batch_dims, -1)
        x = self.after_conv(x)

        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            else:
                carry, x = self.rnn(carry, (x, reset))

        v = self.v_head(x)

        return v, carry


class ACStudentCriticModelMlp(BasicModel):

    conv_encoder: bool = False

    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x

        if self.rnn is not None:
            batch_dims = img.shape[:-1]
            x = self.linear_encoder(img)
            x = x.reshape(*batch_dims, -1)
        else:
            batch_dims = img.shape[:-1]
            x = self.linear_encoder(img)
            x = x.reshape(*batch_dims, -1)

        # NOTE: Continue here tomorrow
        # Is x reshape of shape zero??
        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            elif self.recurrent_arch == 'transformer':
                x = self.rnn(carry, (x, mask, reset))
            else:
                carry, x = self.rnn(carry, (x, reset))

        v = self.v_head(x)

        return v, carry


class ACStudentModel(BasicModel):
    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x

        if self.rnn is not None:
            batch_dims = img.shape[:-3]
            x = self.conv(img)
            x = x.reshape(*batch_dims, -1)
        else:
            batch_dims = img.shape[:-3]
            x = self.conv(img)
            x = x.reshape(*batch_dims, -1)

        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            elif self.recurrent_arch == 'transformer':
                x = self.rnn(carry, (x, mask, reset))
            else:
                carry, x = self.rnn(carry, (x, reset))

        v = self.v_head(x)

        logits = self.pi_head(x)

        return v, logits, carry


class ACTeacherModel(BasicModel):
    """
    Original teacher model from Dennis et al, 2020. It is identical ins
    high-level spec to the student model, but with the additional fwd logic
    of concatenating a noise vector.
    """

    def __call__(self, x, carry=None, reset=None):
        """
        Inputs:
                x: B x h x w observations
                hxs: B x hx_dim hidden states
                masks: B length vector of done masks
        """
        img = x['image']
        time = x['time']
        noise = x.get('noise')
        aux = x.get('aux')

        if self.rnn is not None:
            batch_dims = img.shape[:2]
            x = self.conv(img).reshape(*batch_dims, -1)
        else:
            batch_dims = img.shape[:1]
            x = self.conv(img).reshape(*batch_dims, -1)

        if self.fc_scalar is not None:
            if self.n_scalar_embeddings == 0:
                time /= self.max_scalar

            scalar_emb = self.fc_scalar(time).reshape(*batch_dims, -1)
            x = jnp.concatenate([x, scalar_emb], axis=-1)

        if noise is not None:
            noise = noise.reshape(*batch_dims, -1)
            x = jnp.concatenate([x, noise], axis=-1)

        if aux is not None:
            x = jnp.concatenate([x, aux], axis=-1)

        if self.rnn is not None:
            if self.recurrent_arch == 's5':
                x = self.embed_pre_s5(x)
                carry, x = self.rnn(carry, x, reset)
            else:
                carry, x = self.rnn(carry, (x, reset))

        logits = self.pi_head(x)

        v = self.v_head(x)

        return v, logits, carry


# Register models
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register(
    env_group_id='Overcooked', model_id='default_student_actor_moe',
    entry_point=module_path + ':ACStudentActorModelSoftMoE')

register(
    env_group_id='Overcooked', model_id='default_student_critic_moe',
    entry_point=module_path + ':ACStudentCriticModelMoE')

register(
    env_group_id='Overcooked', model_id='default_student_actor_cnn',
    entry_point=module_path + ':ACStudentActorModel')

register(
    env_group_id='Overcooked', model_id='default_student_critic_cnn',
    entry_point=module_path + ':ACStudentCriticModel')

register(
    env_group_id='Overcooked', model_id='default_student_actor_mlp',
    entry_point=module_path + ':ACStudentActorModelMlp')

register(
    env_group_id='Overcooked', model_id='default_student_critic_mlp',
    entry_point=module_path + ':ACStudentCriticModelMlp')

register(
    env_group_id='Overcooked', model_id='default_student_cnn',
    entry_point=module_path + ':ACStudentModel')

register(
    env_group_id='Overcooked', model_id='default_teacher_cnn',
    entry_point=module_path + ':ACTeacherModel')
