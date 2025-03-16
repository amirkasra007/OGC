"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax
import chex
from typing import Tuple, Union, Optional
from functools import partial
from flax import struct

import jax.numpy as jnp

from .environment import EnvParams, EnvState, Environment


@struct.dataclass
class UEDEnvState:
    encoding: chex.Array
    time: int
    terminal: bool


class UEDEnvironment:
    """
    Wraps two Environment instances, one being the basic environment,
    and the other, its UED counterpart.

    The interface extends the student environment interace.
    """

    def __init__(self, env, ued_env):
        self.env = env
        self.ued_env = ued_env

        # Default reset and step centers on student
        self.reset = self.reset_random
        self.step = self.env.step

    def reset_random(
            self,
            rng: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        return self.env.reset(rng)

    def get_monitored_metrics(self):
        return self.env.get_monitored_metrics() + self.ued_env.get_monitored_metrics()

    def reset_teacher(
            self,
            rng: chex.PRNGKey,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        return self.ued_env.reset(rng)

    def step_teacher(
            self,
            rng: chex.PRNGKey,
            ued_state: EnvState,
            action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        return self.ued_env.step(
            rng, ued_state, action, reset_on_done=False)

    def reset_student(
            self,
            rng: chex.PRNGKey,
            ued_state: EnvState,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """
        Reset the student based on 
        """
        # ued_state_ = UEDEnvState(
        #     encoding=jnp.array([17,  6,  3, 23,  4, 21,  2,  3, 16, 12,  9], dtype=jnp.uint32), time=jnp.array(11, dtype=jnp.uint32), terminal=jnp.array(True))
        encoding = self.ued_env.get_env_instance(rng, ued_state)
        env = self.env.set_env_instance(encoding)
        return env

    def step_student(
            self,
            rng: chex.PRNGKey,
            state: EnvState,
            action: Union[int, float],
            reset_state: Optional[chex.ArrayTree] = None
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        return self.env.step(
            rng,
            state,
            action,
            reset_state=reset_state)

    def set_env_instance(self, encoding: chex.ArrayTree):
        return self.env.set_env_instance(encoding)

    # Spaces interface
    def action_space(self):
        """Action space of the environment."""
        return self.env.action_space()

    def observation_space(self):
        """Observation space of the environment."""
        return self.env.observation_space()

    def state_space(self):
        """Observation space of the environment."""
        return self.env.state_space()

    def max_episode_steps(self):
        """Action space of the environment."""
        return self.env.max_episode_steps()

    def ued_action_space(self):
        """Action space of the environment."""
        return self.ued_env.action_space()

    def ued_observation_space(self):
        """Observation space of the environment."""
        return self.ued_env.observation_space()

    def ued_state_space(self):
        """Observation space of the environment."""
        return self.ued_env.state_space()

    def ued_max_episode_steps(self):
        """Action space of the environment."""
        return self.ued_env.max_episode_steps()

    def get_env_metrics(self, state: EnvState):
        """Environment-specific metrics, e.g. number of walls."""
        return self.env.get_env_metrics(state)

    @property
    def agents(self) -> str:
        """Environment name."""
        return self.env.agents

    @property
    def name(self) -> str:
        """Environment name."""
        return self.env.name
