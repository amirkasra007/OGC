"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
import os
import time

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


from .eval_runner import EvalRunner
from .dr_runner import DRRunner
from .paired_runner import PAIREDRunner
from .plr_runner import PLRRunner
import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.envs.viz.overcooked_visualizer import OvercookedVisualizer


class RunnerInfo:
    def __init__(
            self,
            runner_cls,
            is_ued=False):
        self.runner_cls = runner_cls
        self.is_ued = is_ued


RUNNER_INFO = {
    'dr': RunnerInfo(
        runner_cls=DRRunner,
    ),
    'plr': RunnerInfo(
        runner_cls=PLRRunner,
    ),
    'paired': RunnerInfo(
        runner_cls=PAIREDRunner,
        is_ued=True
    )
}


class ExperimentRunner:
    def __init__(
            self,
            train_runner,
            env_name,
            agent_rl_algo,
            student_model_name,
            student_critic_model_name,
            student_agent_kind="mappo",
            teacher_model_name=None,
            train_runner_kwargs={},
            env_kwargs={},
            ued_env_kwargs={},
            student_rl_kwargs={},
            teacher_rl_kwargs={},
            student_model_kwargs={},
            teacher_model_kwargs={},
            eval_kwargs={},
            eval_env_kwargs={},
            n_devices=1,
            shaped_reward_steps=0,
            shaped_reward_n_updates=0,
            xpid=None
    ):
        self.env_name = env_name
        self.agent_rl_algo = agent_rl_algo
        self.is_ued = RUNNER_INFO[train_runner].is_ued
        self.xpid = xpid
        self.shaped_reward_steps = shaped_reward_steps
        self.shaped_reward_n_updates = shaped_reward_n_updates

        dummy_env = envs.make(
            env_name,
            env_kwargs,
            ued_env_kwargs)[0]

        # ---- Make agent ----
        if student_agent_kind == 'mappo':
            student_model_kwargs['output_dim'] = dummy_env.action_space().n
            student_actor = models.make(
                env_name=env_name,
                model_name=student_model_name,
                **student_model_kwargs
            )

            student_model_kwargs['output_dim'] = 1
            student_critic = models.make(
                env_name=env_name,
                model_name=student_critic_model_name,
                **student_model_kwargs
            )

            student_agent = agents.MAPPOAgent(
                actor=student_actor,
                critic=student_critic,
                n_devices=n_devices,
                **student_rl_kwargs
            )
        else:
            raise ValueError(
                f"Unknown student agent kind: {student_agent_kind}")

        # ---- Handle UED-related settings ----
        if self.is_ued:
            max_teacher_steps = dummy_env.ued_max_episode_steps()
            teacher_model_kwargs['n_scalar_embeddings'] = max_teacher_steps
            teacher_model_kwargs['max_scalar'] = max_teacher_steps
            teacher_model_kwargs['output_dim'] = dummy_env.ued_action_space().n

            teacher_model = models.make(
                env_name=env_name,
                model_name=teacher_model_name,
                **teacher_model_kwargs
            )

            teacher_agent = agents.PPOAgent(
                model=teacher_model,
                n_devices=n_devices,
                **teacher_rl_kwargs
            )

            train_runner_kwargs.update(dict(
                teacher_agents=[teacher_agent]
            ))
            train_runner_kwargs.update(dict(
                ued_env_kwargs=ued_env_kwargs
            ))

        # Debug, tabulate student and teacher model
        # import jax.numpy as jnp
        # dummy_rng = jax.random.PRNGKey(0)
        # obs, _ = dummy_env.reset(dummy_rng)
        # # hx = student_actor.initialize_carry(dummy_rng, (1,))
        # ued_obs, _ = dummy_env.reset_teacher(dummy_rng)
        # # ued_hx = teacher_model.initialize_carry(dummy_rng, (1,))

        # obs['image'] = jnp.expand_dims(obs['image'], 0)
        # ued_obs['image'] = jnp.expand_dims(ued_obs['image'], 0)

        # print(student_actor.tabulate(dummy_rng, obs, None))
        # print(teacher_model.tabulate(dummy_rng, ued_obs, None))

        # import pdb
        # pdb.set_trace()

        # ---- Set up train runner ----
        runner_cls = RUNNER_INFO[train_runner].runner_cls

        # Set up learning rate annealing parameters
        lr_init = train_runner_kwargs.lr
        lr_final = train_runner_kwargs.lr_final
        lr_anneal_steps = train_runner_kwargs.lr_anneal_steps

        if lr_final is None:
            train_runner_kwargs.lr_final = lr_init
        if train_runner_kwargs.lr_final == train_runner_kwargs.lr:
            train_runner_kwargs.lr_anneal_steps = 0

        use_shaped_reward = (shaped_reward_steps is not None and shaped_reward_steps > 0) or (
            shaped_reward_n_updates is not None and shaped_reward_n_updates > 0)

        self.runner = runner_cls(
            env_name=env_name,
            env_kwargs=env_kwargs,
            student_agents=[student_agent],
            student_agent_kind=student_agent_kind,
            n_devices=n_devices,
            shaped_reward=use_shaped_reward,
            **train_runner_kwargs)

        # ---- Make eval runner ----
        if eval_kwargs.get('env_names') is None:
            self.eval_runner = None
        else:
            self.eval_runner = EvalRunner(
                pop=self.runner.student_pop,
                env_kwargs=eval_env_kwargs,
                **eval_kwargs)

        self._start_tick = 0

        # ---- Set up device parallelism ----
        self.n_devices = n_devices
        if n_devices > 1:
            dummy_runner_state = self.reset_train_runner(jax.random.PRNGKey(0))
            self._shmap_run = self._make_shmap_run(dummy_runner_state)
        else:
            self._shmap_run = None

    @partial(jax.jit, static_argnums=(0,))
    def step(self, runner_state, evaluate=False):
        if self.n_devices > 1:
            run_fn = self._shmap_run
        else:
            run_fn = self.runner.run

        stats, *runner_state = run_fn(*runner_state)

        rng = runner_state[0]
        rng, subrng = jax.random.split(rng)

        if self.eval_runner is not None:
            params = runner_state[1].actor_params
            eval_stats = jax.lax.cond(
                evaluate,
                self.eval_runner.run,
                self.eval_runner.fake_run,
                *(subrng, params)
            )
        else:
            eval_stats = {}

        return stats, eval_stats, rng, *runner_state[1:]

    def _make_shmap_run(self, runner_state):
        devices = mesh_utils.create_device_mesh((self.n_devices,))
        mesh = Mesh(devices, axis_names=('device'))

        in_specs, out_specs = self.runner.get_shmap_spec()

        return partial(shard_map,
                       mesh=mesh,
                       in_specs=in_specs,
                       out_specs=out_specs,
                       check_rep=False
                       )(self.runner.run)

    def train(
            self,
            rng,
            agent_algo='ppo',
            algo_runner='dr',
            n_total_updates=1000,
            logger=None,
            log_interval=1,
            test_interval=1,
            checkpoint_interval=0,
            archive_interval=0,
            archive_init_checkpoint=False,
            from_last_checkpoint=False
    ):
        """
        Entry-point for training
        """
        # Load checkpoint if any
        runner_state = self.runner.reset(rng)

        if from_last_checkpoint:
            last_checkpoint_state = logger.load_last_checkpoint_state()
            if last_checkpoint_state is not None:
                runner_state = self.runner.load_checkpoint_state(
                    runner_state,
                    last_checkpoint_state
                )
                self._start_tick = runner_state[1].n_iters[0]

        # Archive initialization weights if necessary
        if archive_init_checkpoint:
            logger.checkpoint(
                self.runner.get_checkpoint_state(runner_state),
                index=0,
                archive_interval=1)

        # Train loop
        log_on = logger is not None and log_interval > 0
        checkpoint_on = checkpoint_interval > 0 or archive_interval > 0
        train_state = runner_state[1]

        tick = self._start_tick
        train_steps = tick*self.runner.step_batch_size * \
            self.runner.n_rollout_steps*self.n_devices
        real_train_steps = train_steps//self.runner.n_students

        while (train_state.n_updates < n_total_updates).any():
            evaluate = test_interval > 0 and (tick+1) % test_interval == 0

            start = time.time()
            stats, eval_stats, *runner_state = \
                self.step(runner_state, evaluate)
            end = time.time()

            start_state = runner_state[-1]
            runner_state = runner_state[:-1]

            if evaluate:
                stats.update(eval_stats)
            else:
                stats.update({k: None for k in eval_stats.keys()})

            train_state = runner_state[1]

            dsteps = self.runner.step_batch_size*self.runner.n_rollout_steps*self.n_devices
            real_dsteps = dsteps//self.runner.n_students
            train_steps += dsteps
            real_train_steps += real_dsteps

            if (self.shaped_reward_steps is not None and self.shaped_reward_steps > 0) or (self.shaped_reward_n_updates is not None and self.shaped_reward_n_updates > 0):
                if self.shaped_reward_n_updates:  # Meassure based on n_updates
                    new_shaped_reward_coeff_value = max(
                        0.0, 1.0 - (train_state.n_updates[0]/self.shaped_reward_n_updates))
                else:  # Meassure based on steps in the env
                    new_shaped_reward_coeff_value = max(
                        0.0, 1.0 - (real_train_steps/self.shaped_reward_steps))

                new_shaped_reward_coeff = jnp.full(
                    runner_state[1].shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value)
                jax.debug.print("Shaped reward coeff: {a}, real_dsteps: {b}, shaped_reward_steps: {c}",
                                a=new_shaped_reward_coeff, b=real_dsteps, c=self.shaped_reward_steps)
                # runner_state[1] is the training state object where the shaped reward coefficient is stored
                runner_state[1] = runner_state[1].set_new_shaped_reward_coeff(
                    new_shaped_reward_coeff)

            sps = int(dsteps/(end-start))
            real_sps = int(real_dsteps/(end-start))
            time_per_iter = float(end-start)
            stats.update(dict(
                steps=train_steps,
                sps=sps,
                real_steps=real_train_steps,
                real_sps=real_sps,
                time_per_iter=time_per_iter,
            ))

            tick += 1

            jax.debug.print("-----\n{stats}", stats=stats)
            if log_on and tick % log_interval == 0:
                logger.log(stats, tick, ignore_val=-np.inf)

            if checkpoint_on and tick > 0:
                if tick % checkpoint_interval == 0 \
                        or (archive_interval > 0 and tick % archive_interval == 0):

                    maze_map = start_state.maze_map
                    agent_dir_idx = start_state.agent_dir_idx
                    agent_inv = start_state.agent_inv
                    for i in range(1):  # self.runner.n_parallel

                        padding = 4  # Fixed
                        grid = np.asarray(
                            maze_map[0, i, padding:-padding, padding:-padding, :])
                        # Render the state
                        frame = OvercookedVisualizer._render_grid(
                            grid,
                            tile_size=32,
                            highlight_mask=None,
                            agent_dir_idx=agent_dir_idx[0][i],
                            agent_inv=agent_inv[0][i]
                        )

                        # Save the numpy frame as image
                        dir = f"{os.getcwd()}/overcooked_teacher_layout_imgs/{self.xpid}/"

                        os.makedirs(os.path.dirname(dir), exist_ok=True)
                        imageio.imwrite(
                            dir + f"{tick}_{i}.png", frame)

                    # Also produce an image of the teachers env output currently
                    checkpoint_state = \
                        self.runner.get_checkpoint_state(runner_state)
                    logger.checkpoint(
                        checkpoint_state,
                        index=tick,
                        archive_interval=archive_interval)
