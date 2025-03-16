from typing import Any

import einops
import jax
import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import constant, orthogonal

from minimax.models.common import StateCNNBase



class MultiExpertLayer(nn.Module):

    in_features: int
    out_features: int
    num_experts: int
    slots_per_expert: int

    def setup(self) -> None:

        self.weight = self.param(
            "weight",
            nn.initializers.xavier_uniform(),
            (self.num_experts, self.in_features, self.out_features),
        )

        self.bias = self.param(
            "bias",
            nn.initializers.xavier_uniform(),
            (self.num_experts, self.out_features),
        )

    def __call__(self, x) -> Any:
        x = einops.einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")

        if self.bias is not None:
            # NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
            # always be 4-dimensional.  But it's easy enough to generalize for 3D
            # inputs as well, so I decided to include that here.
            # if x.ndim == 3:
            #     bias = einops.rearrange(self.bias, "n d -> () n d")
            if x.ndim == 4:
                bias = einops.rearrange(self.bias, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {x.ndim}"
                )
            x = x + bias

        return x


class SoftMoE(nn.Module):

    in_features: int
    out_features: int
    num_experts: int
    slots_per_expert: int

    def setup(self) -> None:
        self.experts = MultiExpertLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            num_experts=self.num_experts,
            slots_per_expert=self.slots_per_expert,

        )
        self.phi = self.param(
            'phi',
            nn.initializers.xavier_uniform(),
            (self.in_features, self.num_experts, self.slots_per_expert),
        )

    def __call__(self, x) -> Any:
        logits = einops.einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = nn.softmax(logits, axis=1)
        # dispatch_weights = logits.softmax(dim=1)  # denoted 'D' in the paper
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = nn.softmax(logits, axis=(-2,-1))
        # combine_weights = einops.rearrange(
        #     nn.softmax(logits.reshape((*logits.shape[:-2], -1)), axis=-1),
        #     # logits.flatten(start_dim=2).softmax(dim=-1),
        #     "b m (n p) -> b m n p",
        #     n=self.num_experts,
        # )

        # NOTE: To save memory, I don't rename the intermediate tensors Y, Ys, Xs.
        # Instead, I just overwrite the 'x' variable.  The names from the paper are
        # included in a comment for each line below.
        x = einops.einsum(
            x, dispatch_weights, "b m d, b m n p -> b n p d")  # Xs
        x = self.experts(x)  # Ys
        x = einops.einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y

        return x


class MoE(nn.Module):
    activation: str = "tanh"
    state_encoder_module: nn.Module = StateCNNBase
    hiddem_dim: int = 64
    recurrent_arch: str = None

    def setup(self) -> None:
        self.state_encoder = self.state_encoder_module(
            activation=self.activation)

        self.moe = SoftMoE(
            in_features=self.state_encoder.out_features,
            out_features=self.hiddem_dim,
            num_experts=4,
            slots_per_expert=32,
        )

        self.proj = nn.Dense(
            self.hiddem_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.proj_layer_norm = nn.LayerNorm()



    def __call__(self, x) -> Any:
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError('Activation not recognized.')
    
        input_shape = x.shape
        state_embedding = self.state_encoder(x)
         
        if len(input_shape) == 5:
            a, n = state_embedding.shape[:2]
            state_embedding = einops.rearrange(state_embedding, "a n ... -> (a n) ...", a=a, n=n)

        state_embedding = einops.rearrange(state_embedding, "... w h c -> ... (w h) c")
        state_embedding = self.moe(state_embedding)
        
        state_embedding = x.reshape((*state_embedding.shape[:-2], -1))

        if len(input_shape) == 5:
            state_embedding = einops.rearrange(state_embedding, "(a n) ... -> a n ...", a=a, n=n)

        state_embedding = self.proj(state_embedding)
        state_embedding = self.proj_layer_norm(state_embedding)
        state_embedding = activation(state_embedding)

        return state_embedding


if __name__ == '__main__':
    rng = jax.random.PRNGKey(30)
    obs = jnp.zeros((200,6,9,26))
    moe = MoE(action_dim=6)
    params = moe.init(rng, obs)
    logits, _ = moe.apply(params, obs)
    jax.debug.breakpoint()