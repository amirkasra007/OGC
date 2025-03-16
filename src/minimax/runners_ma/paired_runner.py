"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from enum import Enum
from functools import partial
from typing import Dict, Tuple, Optional
import inspect

import chex
import einops
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax
import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import minimax.envs as envs
from minimax.util import pytree as _tree_util
from minimax.util.rl import (
    AgentPop,
    VmapTrainState,
    VmapMAPPOTrainState,
    RolloutStorage,
    RolloutStorageSeperate,
    RollingStats,
    UEDScore,
    compute_ued_scores
)


class PAIREDRunner:
    """
    Orchestrates rollouts across one or more students and teachers. 
    The main components at play:
    - AgentPop: Manages train state and batched inference logic 
        for a population of agents.
    - BatchUEDEnv: Manages environment step and reset logic for a 
        population of agents batched over a pair of student and 
        teacher MDPs.
    - RolloutStorage: Manages the storing and sampling of collected txns.
    - PPO: Handles PPO updates, which take a train state + batch of txns.
    """

    def __init__(
        self,
        env_name,
        env_kwargs,
        ued_env_kwargs,
        student_agents,
        student_agent_kind,
        n_students=2,
        n_parallel=1,
        n_eval=1,
        n_rollout_steps=250,
        lr=1e-4,
        lr_final=None,
        lr_anneal_steps=0,
        max_grad_norm=0.5,
        discount=0.99,
        gae_lambda=0.95,
        adam_eps=1e-5,
        teacher_lr=None,
        teacher_lr_final=None,
        teacher_lr_anneal_steps=None,
        teacher_discount=0.99,
        teacher_gae_lambda=0.95,
        teacher_agents=None,
        ued_score='relative_regret',
        track_env_metrics=False,
        n_unroll_rollout=1,
        render=False,
        n_devices=1,
        shaped_reward=False,
    ):
        assert n_parallel % n_devices == 0, 'Num envs must be divisible by num devices.'

        ued_score = UEDScore[ued_score.upper()]

        assert len(student_agents) == 1, \
            'Only one type of student supported.'
        assert not (n_students > 2 and ued_score in [UEDScore.RELATIVE_REGRET, UEDScore.MEAN_RELATIVE_REGRET]), \
            'Standard PAIRED uses only 2 students.'
        assert teacher_agents is None or len(teacher_agents) == 1, \
            'Only one type of teacher supported.'

        self.student_agent_kind = student_agent_kind
        self.n_students = n_students
        self.n_parallel = n_parallel // n_devices
        self.n_eval = n_eval
        self.n_devices = n_devices
        self.step_batch_size = n_students*n_eval*n_parallel
        self.n_rollout_steps = n_rollout_steps
        self.n_updates = 0
        self.lr = lr
        self.lr_final = lr if lr_final is None else lr_final
        self.lr_anneal_steps = lr_anneal_steps
        self.teacher_lr = \
            lr if teacher_lr is None else lr
        self.teacher_lr_final = \
            self.lr_final if teacher_lr_final is None else teacher_lr_final
        self.teacher_lr_anneal_steps = \
            lr_anneal_steps if teacher_lr_anneal_steps is None else teacher_lr_anneal_steps
        self.max_grad_norm = max_grad_norm
        self.adam_eps = adam_eps
        self.ued_score = ued_score
        self.track_env_metrics = track_env_metrics

        self.shaped_reward = shaped_reward

        self.n_unroll_rollout = n_unroll_rollout
        self.render = render

        self.student_pop = AgentPop(student_agents[0], n_agents=n_students)

        if teacher_agents is not None:
            self.teacher_pop = AgentPop(teacher_agents[0], n_agents=1)

        # This ensures correct partial-episodic bootstrapping by avoiding
        # any termination purely due to timeouts.
        # env_kwargs.max_episode_steps = self.n_rollout_steps + 1

        wrappers_lst = ['monitor_return', 'monitor_ep_metrics']
        if self.student_agent_kind == "mappo":
            wrappers_lst = ['world_state_wrapper'] + wrappers_lst

        self.benv = envs.BatchUEDEnv(
            env_name=env_name,
            n_parallel=self.n_parallel,
            n_eval=n_eval,
            env_kwargs=env_kwargs,
            ued_env_kwargs=ued_env_kwargs,
            wrappers=wrappers_lst,
            ued_wrappers=[]
        )
        self.action_dtype = self.benv.env.action_space().dtype

        self.teacher_n_rollout_steps = \
            self.benv.env.ued_max_episode_steps()

        self.student_rollout = RolloutStorageSeperate(
            discount=discount,
            gae_lambda=gae_lambda,
            n_steps=n_rollout_steps,
            n_agents=n_students,
            n_envs=self.n_parallel,
            n_eval=self.n_eval,
            action_space=self.benv.env.action_space(),
            obs_space=self.benv.env.observation_space(),
            obs_space_shared_shape=self.benv.env.world_state_size(),
            agent=self.student_pop.agent
        )

        self.teacher_rollout = RolloutStorage(
            discount=teacher_discount,
            gae_lambda=teacher_gae_lambda,
            n_steps=self.teacher_n_rollout_steps,
            n_agents=1,
            n_envs=self.n_parallel,
            n_eval=1,
            action_space=self.benv.env.ued_action_space(),
            obs_space=self.benv.env.ued_observation_space(),
            agent=self.teacher_pop.agent,
        )

        ued_monitored_metrics = ('return',)
        self.ued_rolling_stats = RollingStats(
            names=ued_monitored_metrics,
            window=10,
        )

        monitored_metrics = self.benv.env.get_monitored_metrics()
        self.rolling_stats = RollingStats(
            names=monitored_metrics,
            window=10,
        )

        self._update_ep_stats = jax.vmap(
            jax.vmap(self.rolling_stats.update_stats))
        self._update_ued_ep_stats = jax.vmap(
            jax.vmap(self.ued_rolling_stats.update_stats))

        if self.render:
            from envs.viz.grid_viz import GridVisualizer
            self.viz = GridVisualizer()
            self.viz.show()

    def reset(self, rng):
        self.n_updates = 0

        n_parallel = self.n_parallel*self.n_devices

        rng, student_rng, teacher_rng = jax.random.split(rng, 3)
        student_info = self._reset_pop(
            student_rng,
            self.student_pop,
            partial(self.benv.reset, sub_batch_size=n_parallel*self.n_eval),
            n_parallel_ep=n_parallel*self.n_eval,
            lr_init=self.lr,
            lr_final=self.lr_final,
            lr_anneal_steps=self.lr_anneal_steps)

        teacher_info = self._reset_teacher_pop(
            teacher_rng,
            self.teacher_pop,
            partial(self.benv.reset_teacher, n_parallel=n_parallel),
            n_parallel_ep=n_parallel,
            lr_init=self.teacher_lr,
            lr_final=self.teacher_lr_final,
            lr_anneal_steps=self.teacher_lr_anneal_steps)

        return (
            rng,
            *student_info,
            *teacher_info
        )

    def _reset_pop(
            self,
            rng,
            pop,
            env_reset_fn,
            n_parallel_ep=1,
            lr_init=3e-4,
            lr_final=3e-4,
            lr_anneal_steps=0):
        rng, *vrngs = jax.random.split(rng, pop.n_agents+1)
        reset_out = env_reset_fn(jnp.array(vrngs))
        if len(reset_out) == 2:
            obs, state = reset_out
        else:
            obs, state, extra = reset_out

        n_parallel = self.n_parallel*self.n_devices

        # dummy_obs = jax.tree_util.tree_map(lambda x: x[0], obs) # for one agent only
        dummy_obs = self._concat_multi_agent_obs(obs)
        dummy_shared_obs = self._concat_multi_agent_obs(obs['world_state'])

        rng, subrng = jax.random.split(rng)
        if self.student_pop.agent.is_recurrent:
            actor_carry, critic_carry = self.student_pop.init_carry(
                subrng, dummy_obs)
            # Technically returns actor and critic carry but we only need one
            self.zero_carry = jax.tree_map(
                lambda x: x.at[:, :self.n_parallel].get(), actor_carry)
        else:
            actor_carry, critic_carry = None, None

        rng, subrng = jax.random.split(rng)
        actor_params, critic_params = self.student_pop.init_params(
            subrng, (dummy_obs[0], dummy_shared_obs[0]))

        schedule_fn = optax.linear_schedule(
            init_value=-float(lr_init),
            end_value=-float(lr_final),
            transition_steps=lr_anneal_steps,
        )

        tx_actor = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.scale_by_adam(eps=self.adam_eps),
            optax.scale_by_schedule(schedule_fn),
        )

        tx_critic = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.scale_by_adam(eps=self.adam_eps),
            optax.scale_by_schedule(schedule_fn),
        )

        shaped_reward_coeff_value = 1.0 if self.shaped_reward else 0.0
        shaped_reward_coeff = jnp.full(
            (self.n_students, 1), fill_value=shaped_reward_coeff_value)
        train_state = VmapMAPPOTrainState.create(
            actor_apply_fn=self.student_pop.agent.evaluate_action,
            actor_params=actor_params,
            actor_tx=tx_actor,
            critic_apply_fn=self.student_pop.agent.get_value,
            critic_params=critic_params,
            critic_tx=tx_critic,
            shaped_reward_coeff=shaped_reward_coeff,
        )

        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(self.n_students, n_parallel*self.n_eval))

        start_state = state

        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(pop.n_agents, n_parallel_ep))

        return train_state, state, obs, actor_carry, critic_carry, ep_stats

    def _reset_teacher_pop(
            self,
            rng,
            pop,
            env_reset_fn,
            n_parallel_ep=1,
            lr_init=3e-4,
            lr_final=3e-4,
            lr_anneal_steps=0):
        rng, *vrngs = jax.random.split(rng, pop.n_agents+1)
        reset_out = env_reset_fn(jnp.array(vrngs))
        if len(reset_out) == 2:
            obs, state = reset_out
        else:
            obs, state, extra = reset_out
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[0], obs)  # for one agent only

        rng, subrng = jax.random.split(rng)
        if pop.agent.is_recurrent:
            carry = pop.init_carry(subrng, obs)
        else:
            carry = None

        rng, subrng = jax.random.split(rng)
        params = pop.init_params(subrng, dummy_obs)

        schedule_fn = optax.linear_schedule(
            init_value=-float(lr_init),
            end_value=-float(lr_final),
            transition_steps=lr_anneal_steps,
        )

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.scale_by_adam(eps=self.adam_eps),
            optax.scale_by_schedule(schedule_fn),
        )

        train_state = VmapTrainState.create(
            apply_fn=pop.agent.evaluate,
            params=params,
            tx=tx
        )

        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(pop.n_agents, n_parallel_ep))

        return train_state, state, obs, carry, ep_stats

    def get_checkpoint_state(self, state):
        _state = list(state)
        _state[1] = state[1].state_dict
        _state[7] = state[7].state_dict

        return _state

    def load_checkpoint_state(self, runner_state, state):
        runner_state = list(runner_state)
        runner_state[1] = runner_state[1].load_state_dict(state[1])
        runner_state[7] = runner_state[7].load_state_dict(state[7])

        return tuple(runner_state)

    @partial(jax.jit, static_argnums=(0, 2,))
    def _get_ma_transition(
        self,
        rng,
        pop,
        actor_params,
        critic_params,
        obs,
        actor_carry,
        critic_carry,
        done
    ):
        ma_obs = self._concat_multi_agent_obs(obs)
        _, pi_params, next_actor_carry = jax.vmap(pop.act, in_axes=(None, 2, 2, None))(
            actor_params, ma_obs, actor_carry, done)
        next_actor_carry = jax.tree_map(lambda x: einops.rearrange(
            x, 't n a d -> a t n d'), next_actor_carry)
        shared_obs = self._concat_multi_agent_obs(obs['world_state'])
        value, next_critic_carry = jax.vmap(pop.get_value, in_axes=(None, 2, 2, None))(
            critic_params, shared_obs, critic_carry, done)
        next_critic_carry = jax.tree_map(lambda x: einops.rearrange(
            x, 't n a d -> a t n d'), next_critic_carry)

        pi = pop.get_action_dist(pi_params, dtype=self.action_dtype)
        rng, subrng = jax.random.split(rng)
        action = pi.sample(seed=subrng)
        log_pi = pi.log_prob(action)

        env_action = {
            'agent_0': action[0],
            'agent_1': action[1]
        }

        # # Add transition to storage
        log_pi = einops.rearrange(log_pi, 'a s n -> s n a')
        value = einops.rearrange(value, 'a s n -> s n a')
        action = einops.rearrange(action, 'a s n -> s n a')

        return (
            shared_obs,
            value,
            log_pi,
            env_action,
            action,
            (jax.tree_map(lambda x: einops.rearrange(
                x, 'n a s d -> s n a d'), next_actor_carry),
             jax.tree_map(lambda x: einops.rearrange(
                 x, 'n a s d -> s n a d'), next_critic_carry)),
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _get_transition(
            self,
            rng,
            pop,
            rollout_mgr,
            rollout,
            params,
            state,
            obs,
            carry,
            done,
            reset_state=None,
            extra=None):
        # Sample action
        if type(params) == tuple and len(params) == 2:
            actor_carry, critic_carry = carry
            actor_params, critic_params = params
            shared_obs, value, log_pi, env_action, action, next_carry = self._get_ma_transition(
                rng,
                pop,
                actor_params,
                critic_params,
                obs,
                actor_carry,
                critic_carry,
                done
            )
            is_multi_agent = True
        else:
            value, pi_params, next_carry = pop.act(params, obs, carry, done)
            pi = pop.get_action_dist(pi_params, dtype=self.action_dtype)
            rng, subrng = jax.random.split(rng)
            env_action = pi.sample(seed=subrng)
            action = env_action  # Is the same in single agent case but a dict in multi_agent
            log_pi = pi.log_prob(action)
            is_multi_agent = False
            shared_obs = None

        rng, *vrngs = jax.random.split(rng, pop.n_agents+1)

        if pop is self.student_pop:
            step_fn = self.benv.step_student
        else:
            step_fn = self.benv.step_teacher
        step_args = (jnp.array(vrngs), state, env_action)

        if reset_state is not None:  # Needed for student to reset to same instance
            step_args += (reset_state,)

        if extra is not None:
            step_args += (extra,)
            next_obs, next_state, reward, done, info, extra = step_fn(
                *step_args)
        else:
            next_obs, next_state, reward, done, info = step_fn(*step_args)

        if is_multi_agent:
            obs = self._concat_multi_agent_obs(obs)

        # Add transition to storage
        if shared_obs is not None:
            done_ = jnp.concatenate(
                [done[:, :, jnp.newaxis], done[:, :, jnp.newaxis]], axis=2)
            # jax.debug.print("info r {i}, info sr {s}, r {r}", i=jnp.sum(
            #     info["sparse_reward"]), s=jnp.sum(info["shaped_reward"]), r=jnp.sum(reward))

            step = (obs, shared_obs, action,
                    info["sparse_reward"], info["shaped_reward"], done_, log_pi, value)
        else:
            step = (obs, action, reward, done, log_pi, value)

        if is_multi_agent:
            if carry[0] is not None:
                step += (carry[0], carry[1])  # Actor and Critic
        else:
            if carry is not None:
                step += (carry,)

        rollout = rollout_mgr.append(rollout, *step)

        if self.render and pop is self.student_pop:
            self.viz.render(
                self.benv.env.env.params,
                jax.tree_util.tree_map(lambda x: x[0][0], state))

        return rollout, next_state, next_obs, next_carry, done, info, extra

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def _rollout(
            self,
            rng,
            pop,
            rollout_mgr,
            n_steps,
            params,
            state,
            obs,
            carry,
            done,
            reset_state=None,
            extra=None,
            ep_stats=None):
        rngs = jax.random.split(rng, n_steps)

        rollout = rollout_mgr.reset()

        def _scan_rollout(scan_carry, rng):
            (rollout,
             state,
             obs,
             carry,
             done,
             extra,
             ep_stats) = scan_carry

            next_scan_carry = \
                self._get_transition(
                    rng,
                    pop,
                    rollout_mgr,
                    rollout,
                    params,
                    state,
                    obs,
                    carry,
                    done,
                    reset_state,
                    extra)

            (rollout,
             next_state,
             next_obs,
             next_carry,
             done,
             info,
             extra) = next_scan_carry

            if ep_stats is not None:
                _ep_stats_update_fn = self._update_ep_stats \
                    if pop is self.student_pop else self._update_ued_ep_stats

                ep_stats = _ep_stats_update_fn(ep_stats, done, info)

            return (rollout, next_state, next_obs, next_carry, done, extra, ep_stats), None

        (rollout, state, obs, carry, done, extra, ep_stats), _ = jax.lax.scan(
            _scan_rollout,
            (rollout, state, obs, carry, done, extra, ep_stats),
            rngs,
            length=n_steps,
            unroll=self.n_unroll_rollout
        )

        return rollout, state, obs, carry, extra, ep_stats

    @partial(jax.jit, static_argnums=(0,))
    def _compile_stats(self,
                       update_stats, ep_stats,
                       ued_update_stats, ued_ep_stats,
                       env_metrics=None,
                       grad_stats=None,
                       ued_grad_stats=None,
                       shaped_reward_coeff=None
                       ):
        mean_returns_by_student = jax.vmap(
            lambda x: x.mean())(ep_stats['return'])
        mean_returns_by_teacher = jax.vmap(
            lambda x: x.mean())(ued_ep_stats['return'])

        mean_ep_stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
            {k: ep_stats[k] for k in self.rolling_stats.names}
        )
        ued_mean_ep_stats = jax.vmap(lambda info: jax.tree_map(lambda x: x.mean(), info))(
            {k: ued_ep_stats[k] for k in self.ued_rolling_stats.names}
        )

        student_stats = {
            f'mean_{k}': v for k, v in mean_ep_stats.items()
        }
        student_stats.update(update_stats)

        stats = {}

        if shaped_reward_coeff is not None:
            stats.update(
                {"shaped_reward_coeff": shaped_reward_coeff})

        for i in range(self.n_students):
            _student_stats = jax.tree_util.tree_map(
                lambda x: x[i], student_stats)  # for agent0
            stats.update({f'{k}_a{i}': v for k, v in _student_stats.items()})

        teacher_stats = {
            f'mean_{k}_tch': v for k, v in ued_mean_ep_stats.items()
        }
        teacher_stats.update({
            f'{k}_tch': v[0] for k, v in ued_update_stats.items()
        })
        stats.update(teacher_stats)

        if self.n_devices > 1:
            stats = jax.tree_map(lambda x: jax.lax.pmean(x, 'device'), stats)

        return stats

    def get_shmap_spec(self):
        runner_state_size = len(inspect.signature(self.run).parameters)
        in_spec = [P(None, 'device'),]*(runner_state_size)
        out_spec = [P(None, 'device'),]*(runner_state_size)

        in_spec[:2] = [P(None),]*2
        in_spec[6] = P(None)
        in_spec = tuple(in_spec)
        out_spec = (P(None),) + in_spec

        return in_spec, out_spec

    @partial(jax.jit, static_argnums=(0,))
    def run(
            self,
            rng,
            train_state,
            state,
            obs,
            actor_carry,
            critic_carry,
            ep_stats,
            ued_train_state,
            ued_state,
            ued_obs,
            ued_carry,
            ued_ep_stats):
        """
        Perform one update step: rollout teacher + students
        """
        if self.n_devices > 1:
            rng = jax.random.fold_in(rng, jax.lax.axis_index('device'))

        # === Reset teacher env + rollout teacher
        rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
        ued_reset_out = self.benv.reset_teacher(jnp.array(vrngs))
        if len(ued_reset_out) > 2:
            ued_obs, ued_state, ued_extra = ued_reset_out
        else:
            ued_obs, ued_state = ued_reset_out
            ued_extra = None

        # Reset UED ep_stats
        if self.ued_rolling_stats is not None:
            ued_ep_stats = self.ued_rolling_stats.reset_stats(
                batch_shape=(1, self.n_parallel))
        else:
            ued_ep_stats = None

        tch_rollout_batch_shape = (1, self.n_parallel*self.n_eval)
        done = jnp.zeros(tch_rollout_batch_shape, dtype=jnp.bool_)
        rng, subrng = jax.random.split(rng)
        ued_rollout, ued_state, ued_obs, ued_carry, _, ued_ep_stats = \
            self._rollout(
                subrng,
                self.teacher_pop,
                self.teacher_rollout,
                self.teacher_n_rollout_steps,
                jax.lax.stop_gradient(ued_train_state.params),
                ued_state,
                ued_obs,
                ued_carry,
                done,
                extra=ued_extra,
                ep_stats=ued_ep_stats
            )

        # === Reset student to new envs + rollout students
        rng, *vrngs = jax.random.split(rng, self.teacher_pop.n_agents+1)
        obs, state, extra = jax.tree_util.tree_map(
            lambda x: x.squeeze(0), self.benv.reset_student(
                jnp.array(vrngs),
                ued_state,
                self.student_pop.n_agents))
        reset_state = state

        # Reset student ep_stats
        st_rollout_batch_shape = (self.n_students, self.n_parallel*self.n_eval)
        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=st_rollout_batch_shape)

        done = jnp.zeros(st_rollout_batch_shape, dtype=jnp.bool_)
        rng, subrng = jax.random.split(rng)
        rollout, state, obs, carry, extra, ep_stats = \
            self._rollout(
                subrng,
                self.student_pop,
                self.student_rollout,
                self.n_rollout_steps,
                (jax.lax.stop_gradient(train_state.actor_params),
                 jax.lax.stop_gradient(train_state.critic_params)),
                state,
                obs,
                (actor_carry, critic_carry),
                done,
                reset_state=reset_state,
                extra=extra,
                ep_stats=ep_stats)

        reward = rollout["rewards"].sum(axis=1).mean(-1)[:, :, jnp.newaxis]
        shaped_reward = rollout["shaped_rewards"].sum(
            axis=1).mean(-1)[:, :, jnp.newaxis]

        ep_stats["reward"] = reward
        ep_stats["shaped_reward"] = shaped_reward
        ep_stats["shaped_reward_scaled_by_shaped_reward_coeff"] = shaped_reward * \
            train_state.shaped_reward_coeff.mean()
        ep_stats["reward_p_shaped_reward_scaled"] = reward + shaped_reward * \
            train_state.shaped_reward_coeff.mean()

        # === Update student with PPO
        # PPOAgent vmaps over the train state and batch. Batch must be N x EM
        _, critic_carry = carry
        shared_obs = self._concat_multi_agent_obs(obs['world_state'])
        value, _ = jax.vmap(self.student_pop.get_value, in_axes=(None, 2, 2))(
            jax.lax.stop_gradient(train_state.critic_params),
            shared_obs,
            critic_carry
        )

        jax.debug.print(
            "train_state.shaped_reward_coeff {s}", s=train_state.shaped_reward_coeff)

        value = einops.rearrange(
            value, "n_env_agents n_students n_parallel -> n_students n_parallel n_env_agents")
        train_batch = self.student_rollout.get_batch(
            rollout,
            value,
            train_state.shaped_reward_coeff
        )

        rng, subrng = jax.random.split(rng)
        train_state, update_stats = self.student_pop.update(
            subrng, train_state, train_batch)

        # === Update teacher with PPO
        # - Compute returns per env per agent
        # - Compute batched returns based on returns per env per agent
        ued_score, _ = compute_ued_scores(
            self.ued_score, train_batch, self.n_eval)
        ued_rollout = self.teacher_rollout.set_final_reward(
            ued_rollout, ued_score)
        ued_train_batch = self.teacher_rollout.get_batch(
            ued_rollout,
            jnp.zeros((1, self.n_parallel))  # Last step terminates episode
        )

        ued_ep_stats = self._update_ued_ep_stats(
            ued_ep_stats,
            jnp.ones((1, len(ued_score), 1), dtype=jnp.bool_),
            {'return': jnp.expand_dims(ued_score, (0, -1))}
        )

        # Update teacher, batch must be 1 x Ex1
        rng, subrng = jax.random.split(rng)
        ued_train_state, ued_update_stats = self.teacher_pop.update(
            subrng, ued_train_state, ued_train_batch)

        # --------------------------------------------------
        # Collect metrics
        if self.track_env_metrics:
            env_metrics = self.benv.get_env_metrics(reset_state)
        else:
            env_metrics = None

        grad_stats, ued_grad_stats = None, None

        stats = self._compile_stats(
            update_stats, ep_stats,
            ued_update_stats, ued_ep_stats,
            env_metrics,
            grad_stats, ued_grad_stats,
            shaped_reward_coeff=train_state.shaped_reward_coeff[0])
        stats.update(dict(n_updates=train_state.n_updates[0]))

        train_state = train_state.increment()
        ued_train_state = ued_train_state.increment()
        self.n_updates += 1

        return (
            stats,
            rng,
            train_state, state, obs, actor_carry, critic_carry, ep_stats,
            ued_train_state, ued_state, ued_obs, ued_carry, ued_ep_stats, reset_state
        )

    def _concat_multi_agent_obs(self, obs: Dict[str, chex.Array]) -> chex.Array:
        """Concatenates a obs dictionary that was built for two env agents.
        Doubles the number of parallel_envs, i.e. (num_students, n_parallel, ...) -> (num_students, 2*n_parallel, ...)
        """
        return jnp.concatenate([obs['agent_0'][:, :, jnp.newaxis, :], obs['agent_1'][:, :, jnp.newaxis, :]], axis=2)

    # def _double_world_state(self, world_state: chex.Array) -> chex.Array:
    #     """Doubles the world state to simulate two agents.
    #     Doubles the number of parallel_envs, i.e. (num_students, n_parallel, ...) -> (num_students, 2*n_parallel, ...)
    #     """
    #     return jnp.concatenate([world_state[:, :, jnp.newaxis, :], world_state[:, :, jnp.newaxis, :]], axis=2)
