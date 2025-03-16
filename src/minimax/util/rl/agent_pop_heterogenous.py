"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


class AgentPopHeterogenous:
    """
    Manages multiple agents with no assumption regarding the architecture
    """

    def __init__(
            self,
            agent_0,
            agent_1,
            n_agents):
        """
        Maintains a set of model parameters.
        """
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.n_agents = n_agents

    def _reshape_to_pop(self, x):
        return jax.tree_map(
            lambda x: jnp.reshape(x, newshape=(self.n_agents, x.shape[0]//self.n_agents, *x.shape[1:])), x)

    def _flatten(self, x):
        return jax.tree_map(lambda x: jnp.reshape(x, newshape=(self.n_agents*x.shape[1], -1)).squeeze(), x)

    def init_params(self, rng, obs):
        if self.agent.is_recurrent:
            # Make time first dim
            obs = jax.tree_map(lambda x: x[jnp.newaxis, :], obs)

        vrngs = jax.random.split(rng, self.n_agents)
        return jax.vmap(
            self.agent.init_params,
            in_axes=(0, None)
        )(vrngs, obs)

    # @partial(jax.jit, static_argnums=(0,))
    # def init_carry(self, rng, obs):
    #     if self.agent.actor.conv_encoder:
    #         agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-3]
    #     else:  # Linear obs
    #         agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-1]
    #     return self.agent.init_carry(rng=rng, batch_dims=agent_batch_dim)

    @partial(jax.jit, static_argnums=(0,))
    def init_carry_agent_0(self, rng, obs):
        if self.agent_0.actor.conv_encoder:
            agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-3]
        else:  # Linear obs
            agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-1]
        return self.agent_0.init_carry(rng=rng, batch_dims=agent_batch_dim)

    @partial(jax.jit, static_argnums=(0,))
    def init_carry_agent_1(self, rng, obs):
        if self.agent_1.actor.conv_encoder:
            agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-3]
        else:  # Linear obs
            agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[:-1]
        return self.agent_1.init_carry(rng=rng, batch_dims=agent_batch_dim)

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, obs, carry, reset=None):
        # If recurrent, add time axis to support scanned rollouts
        actor_0_params, actor_1_params = params
        actor_0_carry, actor_1_carry = carry

        if self.agent_0.is_recurrent:
            # Add time dim after agent dim
            obs_0 = jax.tree_map(
                lambda x: x[:, jnp.newaxis, :], obs['agent_0'])

            if reset is None:
                agent_batch_dim = jax.tree_util.tree_leaves(obs_0)[0].shape[2]
                reset = jnp.zeros(
                    (self.n_agents, 1, agent_batch_dim), dtype=jnp.bool_)
            else:
                reset = reset[:, jnp.newaxis, :]
        else:
            obs_0 = obs['agent_0']

        if self.agent_1.is_recurrent:
            # Add time dim after agent dim
            obs_1 = jax.tree_map(
                lambda x: x[:, jnp.newaxis, :], obs['agent_1'])

            if reset is None:
                agent_batch_dim = jax.tree_util.tree_leaves(obs_1)[0].shape[2]
                reset = jnp.zeros(
                    (self.n_agents, 1, agent_batch_dim), dtype=jnp.bool_)
            else:
                reset = reset[:, jnp.newaxis, :]
        else:
            obs_1 = obs['agent_1']

        value_0, pi_params_0, next_0_carry = jax.vmap(
            self.agent_0.act)(actor_0_params, obs_0, actor_0_carry, reset)

        value_1, pi_param_1, next_1_carry = jax.vmap(
            self.agent_1.act)(actor_1_params, obs_1, actor_1_carry, reset)

        if self.agent_0.is_recurrent:  # Remove time dim
            if value_0 is not None:
                value_0 = value_0.squeeze(1)
            pi_params_0 = jax.tree_map(lambda x: x.squeeze(1), pi_params_0)

        if self.agent_1.is_recurrent:  # Remove time dim
            if value_1 is not None:
                value_1 = value_1.squeeze(1)
            pi_param_1 = jax.tree_map(lambda x: x.squeeze(1), pi_param_1)

        return value_0, value_1, pi_params_0, pi_param_1, next_0_carry, next_1_carry

    def get_action_0_dist(self, dist_params, dtype=jnp.uint8):
        return self.agent_0.get_action_dist(dist_params, dtype=dtype)

    def get_action_1_dist(self, dist_params, dtype=jnp.uint8):
        return self.agent_1.get_action_dist(dist_params, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, obs, carry, reset=None):
        if self.agent.is_recurrent:
            # Add time dim after agent dim
            obs = jax.tree_map(lambda x: x[:, jnp.newaxis, :], obs)

            if reset is None:
                agent_batch_dim = jax.tree_util.tree_leaves(obs)[0].shape[2]
                reset = jnp.zeros(
                    (self.n_agents, 1, agent_batch_dim), dtype=jnp.bool_)
            else:
                reset = reset[:, jnp.newaxis, :]

        value, next_carry = jax.vmap(
            self.agent.get_value)(params, obs, carry, reset)

        if self.agent.is_recurrent:  # Remove time dim
            value = value.squeeze(1)

        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value, next_carry

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def update(self, rng, train_state, batch, prefix_steps=0, fake=False):
        if fake:
            return train_state, jax.vmap(lambda *_: self.agent.get_empty_update_stats())(np.arange(self.n_agents))

        rng, *vrngs = jax.random.split(rng, self.n_agents+1)
        vrngs = jnp.array(vrngs)

        new_train_state, stats = jax.vmap(
            self.agent.update)(vrngs, train_state, batch)
        return new_train_state, stats
