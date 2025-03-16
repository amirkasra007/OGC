


"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from collections import OrderedDict

import einops
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp

from .agent import Agent


class MAPPOAgent(Agent):
    def __init__(
            self,
            actor,
            critic,
            n_epochs=5,
            n_minibatches=1,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            clip_eps=0.2,
            clip_value_loss=True,
            track_grad_norm=False,
            n_unroll_update=1,
            n_devices=1):

        self.actor = actor
        self.critic = critic

        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.clip_value_loss = clip_value_loss
        self.track_grad_norm = track_grad_norm
        self.n_unroll_update = n_unroll_update
        self.n_devices = n_devices

        self.actor_grad_fn = jax.value_and_grad(self._actor_loss, has_aux=True)
        self.critic_grad_fn = jax.value_and_grad(
            self._critic_loss, has_aux=True)

    @property
    def is_recurrent(self):
        # Actor and Critic need to share arch for now.
        return self.actor.is_recurrent

    def init_params(self, rng, obs):
        """
        Returns initialized parameters and RNN hidden state for a specific
        observation shape.
        """
        if len(obs) == 2:
            obs, shared_obs = obs
        else:
            raise ValueError("Obs should always be a two tuple for MAPPO!")

        rng, subrng = jax.random.split(rng)
        is_recurrent = self.actor.is_recurrent
        if is_recurrent:
            batch_size = jax.tree_util.tree_leaves(obs)[0].shape[1]
            actor_carry = self.actor.initialize_carry(
                rng=subrng, batch_dims=(batch_size,))
            critic_carry = self.critic.initialize_carry(
                rng=subrng, batch_dims=(batch_size,))
            reset = jnp.zeros((1, batch_size), dtype=jnp.bool_)

            rng, subrng = jax.random.split(rng)

            # Notice that these are different to later observations but they resemble what we need
            actor_params = self.actor.init(
                subrng, obs[:, :, 0], actor_carry, reset)
            critic_params = self.critic.init(
                subrng, shared_obs[:, :, 0], critic_carry, reset)
        else:

            obs = jnp.concatenate(obs, axis=0)
            shared_obs = jnp.concatenate(shared_obs, axis=0)
            actor_params = self.actor.init(subrng, obs, None)
            critic_params = self.critic.init(subrng, shared_obs, None)

        return (actor_params, critic_params)

    def init_carry(self, rng, batch_dims=1):
        actor_carry = self.actor.initialize_carry(
            rng=rng, batch_dims=batch_dims)
        # This is for evaluation where we throw away the critic
        if self.critic is not None:
            critic_carry = self.critic.initialize_carry(
                rng=rng, batch_dims=batch_dims)
        else:
            critic_carry = None
        return actor_carry, critic_carry

    @partial(jax.jit, static_argnums=(0,))
    def act(self, actor_params, obs, carry=None, reset=None):
        logits, carry = self.actor.apply(
            actor_params, obs, carry, reset)

        return None, logits, carry

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, shared_obs, carry=None, reset=None):
        value, new_carry = self.critic.apply(params, shared_obs, carry, reset)
        return value, new_carry

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_action(
        self, actor_params, action, obs, actor_carry=None, reset=None
    ):
        dist_params, actor_carry = self.actor.apply(
            actor_params, obs, actor_carry, reset)
        dist = self.get_action_dist(dist_params, dtype=action.dtype)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob.squeeze(), \
            entropy.squeeze(), \
            actor_carry

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, action, obs, carry=None, reset=None):
        value, dist_params, carry = self.model.apply(params, obs, carry, reset)
        dist = self.get_action_dist(dist_params, dtype=action.dtype)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return value.squeeze(), \
            log_prob.squeeze(), \
            entropy.squeeze(), \
            carry

    def get_action_dist(self, dist_params, dtype=jnp.uint8):
        return tfp.distributions.Categorical(logits=dist_params, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rng, train_state, batch):
        rngs = jax.random.split(rng, self.n_epochs)

        def _scan_epoch(carry, rng):
            brng, urng = jax.random.split(rng)
            batch, train_state = carry
            minibatches = self._get_minibatches(brng, batch)
            
            
            train_state, stats = \
                self._update_epoch(
                    urng, train_state, minibatches)

            return (batch, train_state), stats

        (_, train_state), stats = jax.lax.scan(
            _scan_epoch,
            (batch, train_state),
            rngs,
            length=len(rngs)
        )

        stats = jax.tree_util.tree_map(lambda x: x.mean(), stats)
        train_state = train_state.increment_updates()

        return train_state, stats

    @partial(jax.jit, static_argnums=(0,))
    def get_empty_update_stats(self):
        keys = [
            'total_loss',  # actor_loss + critic_loss
            'actor_loss',  # loss_actor - entropy_coef*entropy
            'critic_loss',  # value_loss_coef*value_loss
            'actor_loss_actor',  # Without the entropy term added
            'actor_l2_reg_weight_loss',
            'actor_entropy',
            'actor_mean_target',
            'actor_mean_gae',
            'critic_value_loss',
            'critic_l2_reg_weight_loss',
            'critic_mean_value',
            'critic_mean_target',
            'critic_mean_gae',
            'actor_grad_norm',
            'critic_grad_norm',
        ]

        return OrderedDict({k: -jnp.inf for k in keys})

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(
            self,
            rng,
            train_state: TrainState,
            minibatches):

        def _update_minibatch(carry, step):
            rng, minibatch = step
            train_state = carry
            print(type(train_state))
            
            print(type(self))
            (actor_loss, actor_aux_info), actor_grads = self.actor_grad_fn(
                train_state.actor_params,
                train_state.actor_apply_fn,
                minibatch,
                rng,
            )

            (critic_loss, critic_aux_info), critic_grads = self.critic_grad_fn(
                train_state.critic_params,
                train_state.critic_apply_fn,
                minibatch,
                rng,
            )

            total_loss = actor_loss + critic_loss
            loss_info = (total_loss, actor_loss, critic_loss,) + \
                actor_aux_info + critic_aux_info
            loss_info = loss_info + \
                (optax.global_norm(actor_grads), optax.global_norm(critic_grads),)

            if self.n_devices > 1:
                loss_info = jax.tree_map(
                    lambda x: jax.lax.pmean(x, 'device'), loss_info)
                actor_grads = jax.tree_map(
                    lambda x: jax.lax.pmean(x, 'device'), actor_grads)
                critic_grads = jax.tree_map(
                    lambda x: jax.lax.pmean(x, 'device'), critic_grads)

            train_state = train_state.apply_gradients(
                actor_grads=actor_grads,
                critic_grads=critic_grads)

            stats_def = jax.tree_util.tree_structure(OrderedDict({
                k: 0 for k in [
                    'total_loss',  # actor_loss + critic_loss
                    'actor_loss',  # loss_actor - entropy_coef*entropy
                    'critic_loss',  # value_loss_coef*value_loss
                    'actor_loss_actor',  # Without the entropy term added
                    'actor_l2_reg_weight_loss',
                    'actor_entropy',
                    'actor_mean_target',
                    'actor_mean_gae',
                    'critic_value_loss',
                    'critic_l2_reg_weight_loss',
                    'critic_mean_value',
                    'critic_mean_target',
                    'critic_mean_gae',
                    'actor_grad_norm',
                    'critic_grad_norm',
                ]}))

            loss_stats = jax.tree_util.tree_unflatten(
                stats_def, jax.tree_util.tree_leaves(loss_info))
            return train_state, loss_stats

        rngs = jax.random.split(rng, self.n_minibatches)
        train_state, loss_stats = jax.lax.scan(
            _update_minibatch,
            train_state,
            (rngs, minibatches),
            length=self.n_minibatches,
            unroll=self.n_unroll_update
        )

        loss_stats = jax.tree_util.tree_map(
            lambda x: x.mean(axis=0), loss_stats)

        return train_state, loss_stats

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def _actor_loss(
        self,
        params,
        apply_fn,
        batch,
        rng=None
    ):
        """Currently the shape of elements is n_rollout_steps x n_envs x n_env_agents x ...shape.
        This is one more than intended for the actor and critic. The extra dimension is for the
        env agents. We thus need to merge it into the n_envs dimension.
        """
        carry = None

        if self.is_recurrent:
            """
            Elements have batch shape of n_rollout_steps x n_envs//n_minibatches
            """
            batch = jax.tree_map(
                lambda x: einops.rearrange(
                    x, 't n a ... -> t (n a) ...'), batch
            )
            carry = jax.tree_util.tree_map(
                lambda x: x[0, :], batch.actor_carry)
            obs, _, action, rewards, dones, log_pi_old, value_old, target, gae, carry_old, _ = batch

            if self.is_recurrent:
                dones = dones.at[1:, :].set(dones[:-1, :])
                dones = dones.at[0, :].set(False)
                _batch = batch._replace(dones=dones)

                # Returns LxB and LxBxH tensors
                obs, _, action, _, done, _, _, _, _, _, _ = _batch
                log_pi, entropy, carry = apply_fn(
                    params, action, obs, carry, done)
            else:
                log_pi, entropy, carry = apply_fn(
                    params, action, obs, carry_old)
        else:
            batch = jax.tree_map(
                lambda x: einops.rearrange(x, 'n a ... -> (n a) ...'), batch
            )
            obs, _, action, rewards, dones, log_pi_old, value_old, target, gae, _, _ = batch
            log_pi, entropy, _ = apply_fn(params, action, obs, carry)

        ratio = jnp.exp(log_pi - log_pi_old)
        norm_gae = (gae - gae.mean()) / (gae.std() + 1e-5)
        loss_actor1 = ratio * norm_gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps,
                               1.0 + self.clip_eps) * norm_gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        entropy = entropy.mean()

        l2_reg_actor = 0.0

        actor_loss = loss_actor - self.entropy_coef * entropy + l2_reg_actor

        return actor_loss, (
            loss_actor,
            l2_reg_actor,
            entropy,
            target.mean(),
            gae.mean()
        )

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def _critic_loss(
        self,
        params,
        apply_fn,
        batch,
        rng=None
    ):

        carry = None

        if self.is_recurrent:
            """
            Elements have batch shape of n_rollout_steps x n_envs//n_minibatches
            """
            "Same as in actor loss:"
            batch = jax.tree_map(
                lambda x: einops.rearrange(
                    x, 't n a ... -> t (n a) ...'), batch
            )
            carry = jax.tree_util.tree_map(
                lambda x: x[0, :], batch.critic_carry)
            _, obs_shared, action, rewards, dones, log_pi_old, value_old, target, gae, _, carry_old = batch

            if self.is_recurrent:
                dones = dones.at[1:, :].set(dones[:-1, :])
                dones = dones.at[0, :].set(False)
                _batch = batch._replace(dones=dones)

                # Returns LxB and LxBxH tensors
                _, obs_shared, action, _, done, _, _, _, _, _, _ = _batch
                value, carry = apply_fn(
                    params, obs_shared, carry, done)
            else:
                value, carry = apply_fn(
                    params, obs_shared, carry_old)
            value = value.squeeze(-1)
        else:
            batch = jax.tree_map(
                lambda x: einops.rearrange(x, 'n a ... -> (n a) ...'), batch
            )
            obs, obs_shared, action, rewards, dones, log_pi_old, value_old, target, gae, _, _ = batch
            value, _ = apply_fn(params, obs_shared, carry)

        if self.clip_value_loss:
            value_pred_clipped = value_old + (value - value_old).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(value - target)
            value_losses_clipped = jnp.square(value_pred_clipped - target)
            value_loss = 0.5 * \
                jnp.maximum(value_losses, value_losses_clipped).mean()
        else:
            value_pred_clipped = value_old + (value - value_old).clip(
                -self.clip_eps, self.clip_eps
            )
            value_loss = optax.huber_loss(
                value_pred_clipped, target, delta=10.0).mean()

        l2_reg_critic = 0.0

        critic_loss = self.value_loss_coef*value_loss + l2_reg_critic

        return critic_loss, (
            value_loss,
            l2_reg_critic,
            value.mean(),
            target.mean(),
            gae.mean()
        )

    @partial(jax.jit, static_argnums=0)
    def _get_minibatches(self, rng, batch):
        # get dims based on dones
        n_rollout_steps, n_envs = batch.dones.shape[0:2]
        if self.is_recurrent:
            """
            Reshape elements into a batch shape of 
            n_minibatches x n_envs//n_minibatches x n_rollout_steps.
            """
            assert n_envs % self.n_minibatches == 0, \
                'Number of environments must be divisible into number of minibatches.'

            n_env_per_minibatch = n_envs//self.n_minibatches
            shuffled_idx = jax.random.permutation(rng, jnp.arange(n_envs))

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, shuffled_idx, axis=1), batch)

            minibatches = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1).reshape(
                    self.n_minibatches,
                    n_env_per_minibatch,
                    n_rollout_steps,
                    *x.shape[2:]
                ).swapaxes(1, 2), shuffled_batch)
        else:
            n_txns = n_envs*n_rollout_steps
            assert n_envs*n_rollout_steps % self.n_minibatches == 0

            shuffled_idx = jax.random.permutation(rng, jnp.arange(n_txns))
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(
                    x.reshape(n_txns, *x.shape[2:]),
                    shuffled_idx, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(self.n_minibatches, -1, *x.shape[1:]), shuffled_batch)

        return minibatches
