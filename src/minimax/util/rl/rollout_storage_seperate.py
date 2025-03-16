"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp

import minimax.util.pytree as _tree_util
from .ued_scores import compute_episodic_stats


RolloutBatch = namedtuple(
    'RolloutBatch', (
        'obs',
        'obs_shared',
        'actions',
        'rewards',
        'dones',
        'log_pis',
        'values',
        'targets',
        'advantages',
        'actor_carry',
        'critic_carry'
    ))


class RolloutStorageSeperate:
    def __init__(
            self,
            discount,
            gae_lambda,
            n_envs,
            n_eval,
            n_steps,
            action_space,
            obs_space,
            obs_space_shared_shape,
            agent,
            n_agents=1):
        self.discount = discount
        self.gae_lambda = gae_lambda

        # NOTE: n_students refers to minimax's use of n_agents
        # Since I added a multi agent env I need an actual n_agents
        self.n_students = n_agents
        self.n_env_agents = 2
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_evals = n_eval
        self.flat_batch_size = n_envs*self.n_evals
        self.action_space = action_space

        dummy_rng = jax.random.PRNGKey(0)
        self.empty_obs = \
            jax.jax.tree_util.tree_map(
                lambda x: jnp.empty(
                    (self.n_students, n_steps, self.flat_batch_size,
                     self.n_env_agents) + x.shape,
                    dtype=x.dtype
                ),
                obs_space.sample(dummy_rng))

        self.empty_obs_shared = jnp.empty(
            (self.n_students, n_steps, self.flat_batch_size,
             self.n_env_agents) + obs_space_shared_shape)

        self.empty_action = \
            jax.jax.tree_util.tree_map(
                lambda x: jnp.empty(
                    (self.n_students, n_steps, self.flat_batch_size, self.n_env_agents) + x.shape, dtype=x.dtype),
                action_space.sample(dummy_rng))

        if agent.is_recurrent:
            self.empty_actor_carry, self.empty_critic_carry = \
                agent.init_carry(
                    dummy_rng, batch_dims=(
                        self.n_students, n_steps, self.flat_batch_size, self.n_env_agents))
        else:
            self.empty_actor_carry, self.empty_critic_carry = None, None

        if agent.is_recurrent:
            self.append = jax.vmap(self._append_with_carry, in_axes=0)
        else:
            self.append = jax.vmap(self._append_without_carry, in_axes=0)
        self.get_batch = jax.vmap(self._get_batch)
        self.get_return_stats = jax.vmap(
            self._get_return_stats, in_axes=(0, None))

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        """
        Maintains a pytree of rollout transitions and metadata
        """
        if self.empty_actor_carry is None and self.empty_critic_carry is None:
            actor_carry_buffer = None
            critic_carry_buffer = None
        else:
            actor_carry_buffer = self.empty_actor_carry
            critic_carry_buffer = self.empty_critic_carry

        value_batch_size = (self.flat_batch_size,)

        return {
            "obs": self.empty_obs,
            "obs_shared": self.empty_obs_shared,
            "actions": self.empty_action,
            "rewards": jnp.empty(
                (self.n_students, self.n_steps, self.flat_batch_size,
                 self.n_env_agents), dtype=jnp.float32
            ),
            "shaped_rewards": jnp.empty(
                (self.n_students, self.n_steps, self.flat_batch_size,
                 self.n_env_agents), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_students, self.n_steps, self.flat_batch_size,
                                self.n_env_agents), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_students, self.n_steps, self.flat_batch_size,
                 self.n_env_agents), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_students, self.n_steps,
                 *value_batch_size, self.n_env_agents), dtype=jnp.float32
            ),
            "actor_carry": actor_carry_buffer,
            "critic_carry": critic_carry_buffer,
            "_t": jnp.zeros((self.n_students,), dtype=jnp.uint32)  # for vmap
        }

    @partial(jax.jit, static_argnums=0)
    def _append(self, buffer, obs, obs_shared, action, reward, shaped_reward, done, log_pi, value, actor_carry, critic_carry):
        if actor_carry is not None:
            actor_carry_buffer = _tree_util.pytree_set_array_at(
                buffer["actor_carry"], buffer["_t"], actor_carry)
        else:
            actor_carry_buffer = None

        if critic_carry is not None:
            critic_carry_buffer = _tree_util.pytree_set_array_at(
                buffer["critic_carry"], buffer["_t"], critic_carry)
        else:
            critic_carry_buffer = None

        obs = _tree_util.pytree_set_struct_at(buffer["obs"], buffer["_t"], obs)
        obs_shared = _tree_util.pytree_set_struct_at(
            buffer["obs_shared"], buffer["_t"], obs_shared)

        return {
            "obs": obs,
            "obs_shared": obs_shared,
            "actions": _tree_util.pytree_set_struct_at(buffer["actions"], buffer["_t"], action),
            "rewards": buffer["rewards"].at[buffer["_t"]].set(reward.squeeze()),
            "shaped_rewards": buffer["shaped_rewards"].at[buffer["_t"]].set(shaped_reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_t"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_t"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_t"]].set(value.squeeze()),
            "actor_carry": actor_carry_buffer,
            "critic_carry": critic_carry_buffer,
            "_t": (buffer["_t"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def _append_with_carry(self, buffer, obs, obs_shared, action, reward, shaped_reward, done, log_pi, value, actor_carry, critic_carry):
        return self._append(buffer, obs, obs_shared, action, reward, shaped_reward, done, log_pi, value, actor_carry, critic_carry)

    @partial(jax.jit, static_argnums=0)
    def _append_without_carry(self, buffer, obs, obs_shared, action, reward, shaped_reward, done, log_pi, value):
        return self._append(buffer, obs, obs_shared, action, reward, shaped_reward, done, log_pi, value, None, None)

    @partial(jax.jit, static_argnums=(0,))
    # , intrinsic_reward_coeff=0.0):
    def _get_batch(self, buffer, last_value, shaped_reward_coeff=None):
        _dones = buffer["dones"]
        rewards = buffer["rewards"]

        # if intrinsic_reward is not None:
        #     rewards = rewards + 0.0001 * intrinsic_reward_coeff * intrinsic_reward
        # 0.0001 *
        jax.debug.print("rewards buffer {r}", r=rewards.mean())

        rewards = rewards + shaped_reward_coeff.mean() * \
            buffer["shaped_rewards"]

        jax.debug.print("rewards buffer {r}", r=rewards.mean())

        gae, target = self.compute_gae(
            value=buffer["values_old"],
            reward=rewards,
            done=_dones,
            last_value=last_value
        )

        # T x N x E x M --> N x T x EM if recurrent or N x TEM if not
        if self.empty_actor_carry is not None and self.empty_critic_carry is not None:
            actor_carry = buffer["actor_carry"]
            critic_carry = buffer["critic_carry"]
        else:
            actor_carry = None
            critic_carry = None

        batch_kwargs = dict(
            obs=buffer["obs"],
            obs_shared=buffer["obs_shared"],
            actions=buffer["actions"],
            rewards=rewards,
            dones=_dones,
            log_pis=buffer["log_pis_old"],
            values=buffer["values_old"],
            targets=target,
            advantages=gae,
            actor_carry=actor_carry,
            critic_carry=critic_carry
        )
        return RolloutBatch(**batch_kwargs)

    def compute_gae(self, value, reward, done, last_value):
        def _compute_gae(carry, step):
            (discount, gae_lambda, gae, value_next) = carry
            value, reward, done = step

            value_diff = discount*value_next*(1-done) - value
            delta = reward + value_diff

            gae = delta + discount*gae_lambda*(1-done) * gae

            return (discount, gae_lambda, gae, value), gae

        value, reward, done = jnp.flip(value, 0), jnp.flip(
            reward, 0), jnp.flip(done, 0)

        # Handle ensemble values, which have an extra ensemble dim at index -1
        if value.shape != done.shape:
            reward = jnp.expand_dims(reward, -1)
            done = jnp.expand_dims(done, -1)

        gae = jnp.zeros(value.shape[1:])
        _, advantages = jax.lax.scan(
            _compute_gae,
            (self.discount, self.gae_lambda, gae, last_value),
            (value, reward, done),
            length=len(reward)
        )
        advantages = jnp.flip(advantages, 0)
        targets = advantages + jnp.flip(value, 0)

        return advantages, targets

    def _get_return_stats(self, rollout, control_idxs=None):
        if control_idxs is not None:
            positive_signs = (control_idxs == 0)
            reward_signs = -1*(positive_signs.astype(jnp.float32) -
                               (~positive_signs).astype(jnp.float32))
            rewards = rollout["rewards"]*reward_signs
        else:
            rewards = rollout["rewards"]

        pop_batch_shape = (self.n_steps, self.n_envs, self.n_evals)
        rewards = jnp.flip(rewards.reshape(*pop_batch_shape), 0)
        dones = jnp.flip(rollout["dones"].reshape(*pop_batch_shape), 0)

        return compute_episodic_stats(rewards, dones)

    def set_final_reward(self, rollout, reward):
        rollout["rewards"] = rollout["rewards"].at[:, -1, :].set(reward)

        return rollout
