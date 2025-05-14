"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import OrderedDict
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple
import chex
from flax import struct

from minimax.envs.overcooked_proc.overcooked import DIR_TO_VEC, EnvState

from minimax.envs.overcooked_proc.common import OBJECT_TO_INDEX, EnvInstance, make_overcooked_map
from minimax.envs import environment, spaces
from minimax.envs.registration import register_ued


class SequentialActions(IntEnum):
    skip = 0
    wall = 1
    goal = 2
    agent = 3
    onion = 4
    soup = 5
    bowls = 6


@struct.dataclass
class UEDEnvState:
    encoding: chex.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams:
    height: int = 6
    width: int = 9
    n_walls: int = 25
    noise_dim: int = 50
    agent_view_size: int = 5
    replace_wall_pos: bool = False
    fixed_n_wall_steps: bool = False
    first_wall_pos_sets_budget: bool = False
    use_seq_actions: bool = False
    normalize_obs: bool = False
    sample_n_walls: bool = False  # Sample n_walls uniformly in [0, n_walls]
    max_steps: int = 400
    singleton_seed: int = -1
    max_episode_steps: int = 400


class UEDOvercooked(environment.Environment):
    def __init__(
        self,
        height=6,
        width=9,
        n_walls=25,
        noise_dim=16,
        replace_wall_pos=False,
        fixed_n_wall_steps=False,
        first_wall_pos_sets_budget=False,
        use_seq_actions=False,
        normalize_obs=False,
    ):
        """
        Using the original action space requires ensuring proper handling
        of a sequence with trailing dones, e.g. dones: 0 0 0 0 1 1 1 1 1 ... 1.
        Advantages and value losses should only be computed where ~dones[0].
        """
        assert not (first_wall_pos_sets_budget and fixed_n_wall_steps), \
            'Setting first_wall_pos_sets_budget=True requires fixed_n_wall_steps=False.'

        super().__init__()

        self.n_tiles = height*width
        # go straight, turn left, turn right, take action
        self.action_set = jnp.array(jnp.arange(self.n_tiles))

        self.agents = ["agent_0", "agent_1"]

        self.params = EnvParams(
            height=height,
            width=width,
            n_walls=n_walls,
            noise_dim=noise_dim,
            replace_wall_pos=replace_wall_pos,
            fixed_n_wall_steps=fixed_n_wall_steps,
            first_wall_pos_sets_budget=first_wall_pos_sets_budget,
            use_seq_actions=False,
            normalize_obs=normalize_obs,
        )

    @staticmethod
    def align_kwargs(kwargs, other_kwargs):
        kwargs.update(dict(
            height=other_kwargs['height'],
            width=other_kwargs['width'],
        ))

        return kwargs

    def _add_noise_to_obs(self, rng, obs):
        if self.params.noise_dim > 0:
            noise = jax.random.uniform(rng, (self.params.noise_dim,))
            obs.update(dict(noise=noise))

        return obs

    def reset_env(self, key: chex.PRNGKey):
        """
        Prepares the environment state for a new design
        from a blank slate.
        """
        params = self.params
        noise_rng, dir_rng, obj_rng = jax.random.split(key, 3)

        # Initialize encoding with zeros
        encoding = jnp.zeros((self._get_encoding_dim(),), dtype=jnp.uint8)

        # Randomly assign object positions in the encoding
        num_objects = 10  # Adjust this based on max objects
        possible_positions = jnp.arange(self.n_tiles)

        # Randomly select positions for objects
        object_positions = jax.random.choice(obj_rng, possible_positions, shape=(num_objects,), replace=False)

        # Store positions in the encoding array
        encoding = encoding.at[:num_objects].set(object_positions)

        state = UEDEnvState(
            encoding=encoding,
            time=0,
            terminal=False,
        )

        print("Initialized Encoding in reset_env:", encoding)  # Debugging print
        print(f"✅ [UEDOvercooked] Environment reset with params: height={self.params.height}, width={self.params.width}, n_walls={self.params.n_walls}")

        obs = self._add_noise_to_obs(
            noise_rng,
            self.get_obs(state)
        )

        return obs, state


    def get_monitored_metrics(self):
        return ()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: UEDEnvState,
        action: int,
    ) -> Tuple[chex.Array, UEDEnvState, float, bool, dict]:
        """
        Take a design step. 
            action: A pos as an int from 0 to (height*width)-1
        """
        params = self.params

        collision_rng, noise_rng = jax.random.split(key)

        # Sample a random free tile in case of a collision
        dist_values = jnp.logical_and(  # True if position taken
            jnp.ones(params.n_walls + 10),
            jnp.arange(params.n_walls + 10)+1 > state.time
        )

        # Get zero-indexed last wall time step
        if params.fixed_n_wall_steps:
            max_n_walls = params.n_walls
            encoding_pos = state.encoding[:params.n_walls+10]
            last_wall_step_idx = max_n_walls - 1
        else:
            max_n_walls = jnp.round(
                params.n_walls*state.encoding[0]/self.n_tiles).astype(jnp.uint32)

            if self.params.first_wall_pos_sets_budget:
                encoding_pos = state.encoding[:params.n_walls+10]
                last_wall_step_idx = jnp.maximum(max_n_walls, 1) - 1
            else:
                encoding_pos = state.encoding[1:params.n_walls+11]
                last_wall_step_idx = max_n_walls

        pos_dist = jnp.ones(self.n_tiles).at[
            jnp.flip(encoding_pos)].set(jnp.flip(dist_values))
        all_pos = jnp.arange(self.n_tiles, dtype=jnp.uint8)

        agent_step_1_idx = last_wall_step_idx+1  # Enc is full length
        agent_step_2_idx = last_wall_step_idx+2

        # Track whether it is the last time step
        next_state = state.replace(time=state.time + 1)
        done = self.is_terminal(next_state)

        collision = jnp.logical_and(
            pos_dist[action] < 1,
            jnp.logical_or(
                not params.replace_wall_pos,
                jnp.logical_and(  # agent pos cannot be overriden
                    # jnp.logical_or(),
                    # jnp.equal(state.time, goal_step_1_idx),
                    jnp.equal(state.encoding[agent_step_1_idx], action),
                    jnp.equal(state.encoding[agent_step_2_idx], action)
                )
            )
        )
        # collision = (collision * (1-is_agent_dir_step)).astype(jnp.uint32)

        action = (1-collision)*action + \
            collision*jax.random.choice(collision_rng,
                                        all_pos, replace=False, p=pos_dist)

        # (1-is_agent_dir_step)* # + is_agent_dir_step*(-1)
        enc_idx = state.time
        encoding = state.encoding.at[enc_idx].set(action)

        next_state = next_state.replace(
            encoding=encoding,
            terminal=done
        )
        reward = 0

        obs = self._add_noise_to_obs(noise_rng, self.get_obs(next_state))

        # jax.debug.breakpoint()
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(next_state),
            reward,
            done,
            {},
        )

    def get_env_instance(self, key: chex.PRNGKey, state: UEDEnvState) -> chex.Array:
        """
        Converts internal encoding to an instance encoding that 
        can be interpreted by the `set_to_instance` method.
        """
        params = self.params
        h = params.height
        w = params.width
        enc = state.encoding

        # 🛑 Debugging: Print the full encoding array
        # print("🔍 Encoding Array:", enc)

        # Extract object positions from encoding
        if params.fixed_n_wall_steps:
            agent_pos_1_idx = self._get_encoding_dim() - 2
            agent_pos_2_idx = self._get_encoding_dim() - 3
            goal_pos_1_idx = self._get_encoding_dim() - 4
            onion_pos_1_idx = self._get_encoding_dim() - 6
            pot_pos_1_idx = self._get_encoding_dim() - 8
            bowl_pos_1_idx = self._get_encoding_dim() - 10
        else:
            agent_pos_1_idx = jnp.round(params.n_walls*enc[0]/self.n_tiles).astype(jnp.uint32) + 10 - 1
            agent_pos_2_idx = agent_pos_1_idx - 1
            goal_pos_1_idx = agent_pos_1_idx - 3
            onion_pos_1_idx = agent_pos_1_idx - 5
            pot_pos_1_idx = agent_pos_1_idx - 7
            bowl_pos_1_idx = agent_pos_1_idx - 9

        # 🛑 Debugging: Print extracted object positions
        # print(f"📍 Object Positions (Encoded) -> Goal: {enc[goal_pos_1_idx]}, Onion: {enc[onion_pos_1_idx]}, Pot: {enc[pot_pos_1_idx]}, Plate: {enc[bowl_pos_1_idx]}")

        # 🛑 Debugging: Check if indices are valid
        max_grid_idx = h * w - 1
        if enc[goal_pos_1_idx] > max_grid_idx or enc[onion_pos_1_idx] > max_grid_idx or \
        enc[pot_pos_1_idx] > max_grid_idx or enc[bowl_pos_1_idx] > max_grid_idx:
            print("⚠ Warning: Some object indices exceed grid bounds!")
        # === Extract object positions from encoding ===
        if params.fixed_n_wall_steps:
            n_walls = params.n_walls  # Fixed number of walls
            enc_len = self._get_encoding_dim()  # Full encoding length
            wall_pos_idx = jnp.flip(enc[:n_walls])  # Wall positions are in the first `n_walls` elements
        else:
            n_walls = jnp.round(params.n_walls * enc[0] / self.n_tiles).astype(jnp.uint32)
            
            if params.first_wall_pos_sets_budget:
                # Extract the wall positions (with padding)
                wall_pos_idx = jnp.flip(enc[:params.n_walls])
                enc_len = n_walls + 10
            else:
                # Wall positions start from index 1 onwards
                wall_pos_idx = jnp.flip(enc[1:params.n_walls + 1])
                enc_len = n_walls + 11

        # 🛑 Debugging: Print wall positions and number of walls
        # print(f"🧱 Wall Positions (Encoded): {wall_pos_idx}")
        # print(f"🏗️ Number of Walls: {n_walls}")



        # Make wall map
        wall_start_time = jnp.logical_and(  # 1 if explicitly predict # blocks, else 0
            not params.fixed_n_wall_steps,
            not params.first_wall_pos_sets_budget
        ).astype(jnp.uint32)

        wall_map = jnp.zeros((h * w), dtype=jnp.bool_)
        wall_values = jnp.arange(
            params.n_walls) + wall_start_time < jnp.minimum(state.time, n_walls + wall_start_time)
        wall_values = jnp.flip(wall_values)
        wall_map = wall_map.at[wall_pos_idx].set(wall_values)
        wall_map = wall_map.reshape((h, w))
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.reshape(-1)

        occupied_mask = wall_map
        print(f"🛑 Occupied Mask:\n{occupied_mask.reshape(self.params.height, self.params.width)}")

        """Agents should always end up on an empty square. If they are placed on a wall pick randomly."""
        is_occupied = occupied_mask[enc[agent_pos_1_idx]] == 1
        agent_pos_1_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(h*w), shape=(
        ), p=jnp.logical_not(occupied_mask)) + jnp.logical_not(is_occupied)*enc[agent_pos_1_idx]
        agent_1_placed = state.time > jnp.array(
            [agent_pos_1_idx], dtype=jnp.uint8)
        agent_1_pos = \
            agent_1_placed*jnp.array([agent_pos_1_idx_enc % w, agent_pos_1_idx_enc // w], dtype=jnp.uint8) \
            + (~agent_1_placed)*jnp.array([h, w], dtype=jnp.uint8)
        occupied_mask = occupied_mask.at[agent_pos_1_idx_enc].set(True)

        is_occupied = occupied_mask[enc[agent_pos_2_idx]] == 1
        agent_pos_2_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(
            h*w), shape=(), p=jnp.logical_not(occupied_mask)) + jnp.logical_not(is_occupied)*enc[agent_pos_2_idx]
        agent_2_placed = state.time > jnp.array(
            [agent_pos_2_idx], dtype=jnp.uint8)
        agent_2_pos = \
            agent_2_placed*jnp.array([agent_pos_2_idx_enc % w, agent_pos_2_idx_enc // w], dtype=jnp.uint8) \
            + (~agent_2_placed)*jnp.array([h, w], dtype=jnp.uint8)
        occupied_mask = occupied_mask.at[agent_pos_2_idx_enc].set(True)

        agents_obj_occupied_mask = jnp.zeros_like(occupied_mask)
        agents_obj_occupied_mask = agents_obj_occupied_mask.reshape((h, w))
        # Exlude corners, will never be actually reachable
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[0, 0].set(True)
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[0, -1].set(True)
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[-1, 0].set(True)
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[-1, -1].set(
            True)
        agents_obj_occupied_mask = agents_obj_occupied_mask.reshape(-1)
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            agent_pos_1_idx_enc].set(True)
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            agent_pos_2_idx_enc].set(True)

        """Objects can end up on a wall but never on a agent or another agent."""
        is_occupied = agents_obj_occupied_mask[enc[goal_pos_1_idx]] == 1
        goal_pos_1_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(
            h*w), shape=(), p=jnp.logical_not(agents_obj_occupied_mask)) + jnp.logical_not(is_occupied)*enc[goal_pos_1_idx]
        goal_1_placed = jnp.array(True, dtype=jnp.bool_)

        # print(f"🔍 Encoded Goal Position Index: {goal_pos_1_idx}")
        # print(f"📍 Mapped Goal Position (Flat Index): {goal_pos_1_idx_enc}")
        # print(f"✅ Is Occupied Before Placement? {is_occupied}")
        # print(f"⏳ Time Step: {state.time}, Goal Placement Condition: {goal_1_placed}")

        goal_1_pos = \
            goal_1_placed*jnp.zeros((h*w), dtype=jnp.uint8).at[goal_pos_1_idx_enc].set(1) \
            + (~goal_1_placed)*jnp.zeros((h*w), dtype=jnp.uint8)
        goal_1_pos = goal_1_pos.reshape((h, w))
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            goal_pos_1_idx_enc].set(True)
        wall_map = wall_map.at[goal_pos_1_idx_enc].set(True)
        # print(f"🟢 Final Object Grid - Goals:\n{goal_1_pos}")

        # Verify Goal Position
        # print(f"📍 Goal Position Index: {goal_pos_1_idx}, Encoded Value: {enc[goal_pos_1_idx]}, Mapped Position: {goal_pos_1_idx_enc}")
        ##############################

        is_occupied = agents_obj_occupied_mask[enc[onion_pos_1_idx]] == 1
        onion_pos_1_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(
            h*w), shape=(), p=jnp.logical_not(agents_obj_occupied_mask)) + jnp.logical_not(is_occupied)*enc[onion_pos_1_idx]
        # onion_1_placed = state.time > jnp.array(
        #     [onion_pos_1_idx], dtype=jnp.uint8)
        onion_1_placed = jnp.array(True, dtype=jnp.bool_)

        #######################
        # print(f"🔍 Encoded Onion Position Index: {onion_pos_1_idx}")
        # print(f"📍 Mapped Onion Position (Flat Index): {onion_pos_1_idx_enc}")
        # print(f"✅ Is Occupied Before Placement? {is_occupied}")
        # print(f"⏳ Time Step: {state.time}, Onion Placement Condition: {onion_1_placed}")
        #######################
            
        onion_1_pos = \
            onion_1_placed*jnp.zeros((h*w), dtype=jnp.uint8).at[onion_pos_1_idx_enc].set(1) \
            + (~onion_1_placed)*jnp.zeros((h*w), dtype=jnp.uint8)
        onion_1_pos = onion_1_pos.reshape((h, w))
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            onion_pos_1_idx_enc].set(True)
        wall_map = wall_map.at[onion_pos_1_idx_enc].set(True)
        # print(f"🟢 Final Object Grid - Onions:\n{onion_1_pos.reshape(h, w)}")
        # Verify Onion Position
        # print(f"🧅 Onion Position Index: {onion_pos_1_idx}, Encoded Value: {enc[onion_pos_1_idx]}, Mapped Position: {onion_pos_1_idx_enc}")
        #######################################################

        is_occupied = agents_obj_occupied_mask[enc[pot_pos_1_idx]] == 1
        pot_pos_1_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(
            h*w), shape=(), p=jnp.logical_not(agents_obj_occupied_mask)) + jnp.logical_not(is_occupied)*enc[pot_pos_1_idx]
        pot_1_placed = jnp.array(True, dtype=jnp.bool_)

        ###################################################
        # print(f"🔍 Encoded Pot Position Index: {pot_pos_1_idx}")
        # print(f"📍 Mapped Pot Position (Flat Index): {pot_pos_1_idx_enc}")
        # print(f"✅ Is Occupied Before Placement? {is_occupied}")
        # print(f"⏳ Time Step: {state.time}, Pot Placement Condition: {pot_1_placed}")
        #########################################

        pot_1_pos = \
            pot_1_placed*jnp.zeros((h*w), dtype=jnp.uint8).at[pot_pos_1_idx_enc].set(1) \
            + (~pot_1_placed)*jnp.zeros((h*w), dtype=jnp.uint8)
        pot_1_pos = pot_1_pos.reshape((h, w))
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            pot_pos_1_idx_enc].set(True)
        wall_map = wall_map.at[pot_pos_1_idx_enc].set(True)
        # print(f"🟢 Final Object Grid - Pots:\n{pot_1_pos}")


        # Verify Pot Position
        # print(f"🍲 Pot Position Index: {pot_pos_1_idx}, Encoded Value: {enc[pot_pos_1_idx]}, Mapped Position: {pot_pos_1_idx_enc}")


        is_occupied = agents_obj_occupied_mask[enc[bowl_pos_1_idx]] == 1
        bowl_pos_1_idx_enc = is_occupied*jax.random.choice(key, jnp.arange(
            h*w), shape=(), p=jnp.logical_not(agents_obj_occupied_mask)) + jnp.logical_not(is_occupied)*enc[bowl_pos_1_idx]
        bowl_1_placed = jnp.array(True, dtype=jnp.bool_)

        ###################################################
        # print(f"🔍 Encoded Plate Position Index: {bowl_pos_1_idx}")
        # print(f"📍 Mapped Plate Position (Flat Index): {bowl_pos_1_idx_enc}")
        # print(f"✅ Is Occupied Before Placement? {is_occupied}")
        # print(f"⏳ Time Step: {state.time}, Plate Placement Condition: {bowl_1_placed}")
        #########################################

        bowl_1_pos = \
            bowl_1_placed*jnp.zeros((h*w), dtype=jnp.uint8).at[bowl_pos_1_idx_enc].set(1) \
            + (~bowl_1_placed)*jnp.zeros((h*w), dtype=jnp.uint8)
        bowl_1_pos = bowl_1_pos.reshape((h, w))
        agents_obj_occupied_mask = agents_obj_occupied_mask.at[
            bowl_pos_1_idx_enc].set(True)
        wall_map = wall_map.at[bowl_pos_1_idx_enc].set(True)
        # print(f"🟢 Final Object Grid - Plates:\n{bowl_1_pos}")

        # Verify Plate Position
        # print(f"🍽 Plate Position Index: {bowl_pos_1_idx}, Encoded Value: {enc[bowl_pos_1_idx]}, Mapped Position: {bowl_pos_1_idx_enc}")
        #####################debugging####################

        # agent_dir_idx = jnp.floor((4*enc[-1]/self.n_tiles)).astype(jnp.uint8)
        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(
            len(DIR_TO_VEC), dtype=jnp.int32), shape=(2,))

        # Zero out walls where agent and goal reside
        # Should not be the case but just in case
        agent_1_mask = agent_1_placed * \
            (~(jnp.arange(h*w) == agent_pos_1_idx_enc)) + ~agent_1_placed*wall_map
        agent_2_mask = agent_2_placed * \
            (~(jnp.arange(h*w) == agent_pos_2_idx_enc)) + ~agent_2_placed*wall_map
        goal_mask = goal_1_placed * \
            (~(jnp.arange(h*w) == goal_pos_1_idx_enc)) + ~goal_1_placed*wall_map
        wall_map = wall_map*agent_1_mask*agent_2_mask
        wall_map = wall_map.reshape(h, w)

        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        key, subkey = jax.random.split(key)
        random_agent_inv = jax.random.choice(
            subkey, possible_items, shape=(2,), replace=True)

        return EnvInstance(
            agent_pos=jnp.array([agent_1_pos, agent_2_pos], dtype=jnp.uint32),
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_1_pos,
            wall_map=wall_map,
            onion_pile_pos=onion_1_pos,
            pot_pos=pot_1_pos,
            plate_pile_pos=bowl_1_pos,
            agent_inv=random_agent_inv
        )

    def is_terminal(self, state: UEDEnvState) -> bool:
        done_steps = state.time >= self.max_episode_steps()
        return jnp.logical_or(done_steps, state.terminal)

    def _get_post_terminal_obs(self, state: UEDEnvState):
        dtype = jnp.float32 if self.params.normalize_obs else jnp.uint8
        image = jnp.zeros((
            self.params.height+2, self.params.width+2, 3), dtype=dtype
        )

        return OrderedDict(dict(
            image=image,
            time=state.time,
            noise=jnp.zeros(self.params.noise_dim, dtype=jnp.float32),
        ))

    def get_obs(self, state: UEDEnvState, flatten_objects: bool = False):
        instance = self.get_env_instance(jax.random.PRNGKey(0), state)
        h = self.params.height
        w = self.params.width
        onion_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        plate_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        dish_pos = jnp.zeros((h, w), dtype=jnp.uint8)

        pot_status = jnp.ones(
            (instance.wall_map.reshape(-1).shape), dtype=jnp.uint8) * 23

        agent_dir = DIR_TO_VEC.at[instance.agent_dir_idx].get()

        maze_map = make_overcooked_map(
            wall_map=instance.wall_map,
            goal_pos=instance.goal_pos,
            agent_pos=instance.agent_pos,
            agent_dir_idx=instance.agent_dir_idx,
            plate_pile_pos=instance.plate_pile_pos,
            onion_pile_pos=instance.onion_pile_pos,
            pot_pos=instance.pot_pos,
            pot_status=pot_status,
            onion_pos=onion_pos,
            plate_pos=plate_pos,
            dish_pos=dish_pos,
            pad_obs=True,
            num_agents=2,
            agent_view_size=5
        )

        padding = 4
        obj_tensor = self.get_object_representation(state, maze_map[padding:-padding, padding:-padding, :])

        if flatten_objects:
            obj_tensor = obj_tensor.reshape(-1)  # flatten to 1D

        return OrderedDict(dict(
            image=maze_map[padding:-padding, padding:-padding, :],
            time=state.time,
            objects=obj_tensor,
        ))

    @property
    def default_params(self):
        return EnvParams()

    @property
    def name(self) -> str:
        """Environment name."""
        return "UEDOvercooked"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        params = self.params
        return spaces.Discrete(
            params.height*params.width,
            dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        params = self.params
        max_episode_steps = self.max_episode_steps()
        spaces_dict = {
            'image': spaces.Box(0, 255, (params.height, params.width, 3)),
            'time': spaces.Discrete(max_episode_steps),
        }
        if self.params.noise_dim > 0:
            spaces_dict.update({
                'noise': spaces.Box(0, 1, (self.params.noise_dim,))
            })
        return spaces.Dict(spaces_dict)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        params = self.params
        encoding_dim = self._get_encoding_dim()
        max_episode_steps = self.max_episode_steps()
        h = params.height
        w = params.width
        return spaces.Dict({
            'encoding': spaces.Box(0, 255, (encoding_dim,)),
            'time': spaces.Discrete(max_episode_steps),
            "terminal": spaces.Discrete(2),
        })

    def _get_encoding_dim(self) -> int:
        encoding_dim = self.max_episode_steps()
        # if not self.params.set_agent_dir:
        #     encoding_dim += 1 # max steps is 1 less than full encoding dim

        return encoding_dim

    def max_episode_steps(self) -> int:
        if self.params.fixed_n_wall_steps \
                or self.params.first_wall_pos_sets_budget:
            max_episode_steps = self.params.n_walls + 10
        else:
            max_episode_steps = self.params.n_walls + 11

        return max_episode_steps
    
    def get_max_objects(self):
        """Determine the maximum number of each object type in the environment."""

        # Reset the environment and get an initial state
        state = self.reset_env(jax.random.PRNGKey(0))[1]

        # Get environment instance representation
        instance = self.get_env_instance(jax.random.PRNGKey(0), state)

        # Debugging: Print instance object positions
        print("Instance Object Positions:")
        print(f"Onion Piles:\n{instance.onion_pile_pos}")
        print(f"Pots:\n{instance.pot_pos}")
        print(f"Plates:\n{instance.plate_pile_pos}")
        print(f"Goals:\n{instance.goal_pos}")

        # Compute maximum number of each object type
        max_onions = jnp.sum(instance.onion_pile_pos)    # Count of onion piles
        max_pots = jnp.sum(instance.pot_pos)            # Count of cooking pots
        max_plates = jnp.sum(instance.plate_pile_pos)   # Count of plate piles
        max_goals = jnp.sum(instance.goal_pos)          # Count of goal locations

        total_objects = max_onions + max_pots + max_plates + max_goals

        print(f"Max Objects -> Onions: {max_onions}, Pots: {max_pots}, Plates: {max_plates}, Goals: {max_goals}")

        return total_objects  # Total number of objects in the environment

    def get_object_representation(self, state: UEDEnvState, maze_map: jnp.ndarray ) -> jnp.ndarray:
        """Returns a dynamic, agent-friendly object-centric representation tensor."""

        instance = self.get_env_instance(jax.random.PRNGKey(0), state)
        h, w = self.params.height, self.params.width

        # === Extract object layer ===
        object_layer = maze_map[..., 0]  # maze_map: [h+pad, w+pad, 3] → object types in [:, :, 0]

        # === Core object maps ===
        onion_map = instance.onion_pile_pos
        pot_map = instance.pot_pos
        plate_map = instance.plate_pile_pos
        goal_map = instance.goal_pos

        # === Agent position maps ===
        agent_maps = []
        for i in range(2):  # Assuming 2 agents
            agent_pos = instance.agent_pos[i]
            agent_map = jnp.zeros((h, w), dtype=jnp.uint8).at[agent_pos[1], agent_pos[0]].set(1)
            agent_maps.append(agent_map)

        # === Wall map ===
        wall_map = instance.wall_map.astype(jnp.uint8)

        # === Dynamic loose items ===
        loose_onion_map = (object_layer == OBJECT_TO_INDEX['onion']).astype(jnp.uint8)
        loose_plate_map = (object_layer == OBJECT_TO_INDEX['plate']).astype(jnp.uint8)
        loose_dish_map  = (object_layer == OBJECT_TO_INDEX['dish']).astype(jnp.uint8)


        # === Stack all features into a single tensor ===
        object_tensor = jnp.stack([
            onion_map,        # [h, w]
            pot_map,          # [h, w]
            plate_map,        # [h, w]
            goal_map,         # [h, w]
            agent_maps[0],    # [h, w]
            agent_maps[1],    # [h, w]
            wall_map,          # [h, w]
            loose_onion_map,   # new
            loose_plate_map,   # new
            loose_dish_map     # new
        ], axis=-1)  # [h, w, 7]

        print(f"Final Agent-Friendly Object Tensor Shape: {object_tensor.shape}")
        print(f"Agent-Friendly Object Tensor: {object_tensor}")
        return object_tensor

if __name__ == "__main__":
    env = UEDOvercooked()
    rng = jax.random.PRNGKey(42)

    # Reset environment and get initial state
    obs, state = env.reset_env(rng)

    obj_grid = env.get_object_representation(state)
    print("Unified Object Grid Shape:", obj_grid.shape)

    obj_types = ['onion', 'pot', 'plate', 'goal']
    for i, name in enumerate(obj_types):
        print(f"🗺️ {name.capitalize()} Map:\n{obj_grid[:, :, i]}")

if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register_ued(env_id='Overcooked', entry_point=module_path + ':UEDOvercooked')