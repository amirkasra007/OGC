from functools import partial
import jax
import jax.numpy as jnp

import chex
from typing import Union, Optional

from minimax.envs.environment import EnvState

from minimax.envs import environment
from minimax.envs.wrappers.env_wrapper import EnvWrapper


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers.
    Copied from the JaxMARL project: https://github.com/FLAIROx/JaxMARL
    """

    def __init__(self, env: environment.Environment):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


class WorldStateWrapper(EnvWrapper):

    def __init__(self, env):
        self._env = env

        self._wrap_level = 1
        while hasattr(env, '_env'):
            if isinstance(env, EnvWrapper):
                self._wrap_level += 1

            env = env._env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]

        NOTE: This assumes two agents!
        """
        # This is consistent with the OvercookedEnv implementation.
        world_state_0 = jnp.concatenate(
            [obs['agent_0'], obs['agent_1']], axis=-1)
        world_state_1 = jnp.concatenate(
            [obs['agent_1'], obs['agent_0']], axis=-1)

        return {
            'agent_0': world_state_0,
            'agent_1': world_state_1
        }

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        res = self._env.reset(key)
        obs = res[0]
        world_state = self.world_state(obs)
        obs["world_state"] = world_state
        _tuple = (obs, *res[1:])
        return self._append_extra_to_tuple(_tuple)

    @partial(jax.jit, static_argnums=0)
    def step(self,
             key: chex.PRNGKey,
             state: EnvState,
             action: Union[int, float],
             reset_state: Optional[chex.ArrayTree] = None,
             extra: dict = None,
             **kwargs):
        if self._wrap_level > 1:
            obs, env_state, reward, done, info = self._env.step(
                key, state, action, **kwargs
            )
            world_state = self.world_state(obs)
            obs["world_state"] = world_state
            return obs, env_state, reward, done, info
        else:
            obs, env_state, reward, done, info = self._env.step(
                key, state, action, **kwargs
            )
            world_state = self.world_state(obs)
            obs["world_state"] = world_state
            _tuple = (obs, env_state, reward, done, info)
            return self._append_extra_to_tuple(_tuple, extra)

    @partial(jax.jit, static_argnums=0)
    def set_state(self, state):
        if self._wrap_level > 1:
            obs, state = self._env.set_state(state)
            world_state = self.world_state(obs)
            obs["world_state"] = world_state
            return obs, state
        else:
            obs, state = self._env.set_state(state)
            world_state = self.world_state(obs)
            obs["world_state"] = world_state
            _tuple = (obs, state)
            return self._append_extra_to_tuple(_tuple)

    @partial(jax.jit, static_argnums=0)
    def reset_student(
        self,
        key,
        state
    ):
        res = self._env.reset_student(key, state)
        obs = res[0]
        world_state = self.world_state(obs)
        obs["world_state"] = world_state
        return obs, *res[1:]

    def world_state_size(self):
        spaces = [
            jnp.zeros(self._env.observation_space().shape) for _ in self._env.agents]
        y = jnp.concatenate(spaces, axis=-1).shape
        return y

    def reset_extra(self):
        if self._wrap_level > 1:
            extra = self._env.reset_extra()
        else:
            extra = {}
        return extra

    def reset_teacher(
        self,
        rng
    ):
        _tuple = self._env.reset_teacher(rng)

        return self._append_extra_to_tuple(_tuple)

    def step_teacher(
        self,
        rng,
        ued_state,
        action,
        extra: dict = None,
    ):
        if self._wrap_level > 1:
            return self._env.step_teacher(rng, ued_state, action, extra)
        else:
            _tuple = self._env.step_teacher(rng, ued_state, action)
            return self._append_extra_to_tuple(_tuple)

    @classmethod
    def is_compatible(cls, env):
        return env.name == "Overcooked"
