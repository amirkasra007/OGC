# Edited from JaxMarl: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked

from enum import IntEnum
from hmac import new
import os
import random
import time

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict
import chex
from flax import struct
from sympy import jn

from minimax.envs import environment, spaces
from minimax.envs.registration import register
from minimax.envs.overcooked_proc.layouts import layout_grid_to_onehot_dict
import minimax.util.graph as _graph_util
from minimax.envs.viz.overcooked_visualizer import OvercookedVisualizer

from .common import EnvInstance, make_overcooked_map

asymm_advantages_6_9 = """
WWWWWWWWW
O WXWOW X
W   P A W
WA  P   W
WWWBWBWWW
WWWWWWWWW
"""

counter_circuit_6_9 = """
WWWPPWWWW
W A    WW
B WWWW XW
W     AWW
WWWOOWWWW
WWWWWWWWW
"""

forced_coord_6_9 = """
WWWPWWWWW
OAWAPWWWW
O W WWWWW
B W WWWWW
WWWXWWWWW
WWWWWWWWW
"""

cramped_room_6_9 = """
WWPWWWWWW
OAA OWWWW
W   WWWWW
WBWXWWWWW
WWWWWWWWW
WWWWWWWWW
"""

coord_ring_6_9 = """
WWWPWWWWW
WA APWWWW
B W WWWWW
O   WWWWW
WOXWWWWWW
WWWWWWWWW
"""

forced_coord_5_5 = """
WWWPW
OAWAP
O W W
B W W
WWWXW
"""

cramped_room_5_5 = """
WWPWW
OAA O
W   W
WBWXW
WWWWW
"""

coord_ring_5_5 = """
WWWPW
WA AP
B W W
O   W
WOXWW
"""


class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
    done = 6


@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    bowl_pile_pos: chex.Array
    onion_pile_pos: chex.Array
    time: int
    terminal: bool

@struct.dataclass
class EnvParams:
    height: int = 6
    width: int = 9
    h_min: int = 4
    w_min: int = 4
    n_walls: int = 5
    agent_view_size: int = 5
    replace_wall_pos: bool = False
    normalize_obs: bool = False
    sample_n_walls: bool = False  # Sample n_walls uniformly in [0, n_walls]
    max_steps: int = 400
    singleton_seed: int = -1
    max_episode_steps: int = 400


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23  # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
# 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_FULL_STATUS = 20
POT_READY_STATUS = 0
# A pot has at most 3 onions. A soup contains exactly 3 onions.
MAX_ONIONS_IN_POT = 3

URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


SHAPED_REWARD = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "PICKUP_TOMATO_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

OBJECT_TO_INDEX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "onion": 3,
    "onion_pile": 4,
    "plate": 5,
    "plate_pile": 6,
    "goal": 7,
    "pot": 8,
    "dish": 9,
    "agent": 10,
}


COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'white': np.array([255, 255, 255]),
    'black': np.array([25, 25, 25]),
    'orange': np.array([230, 180, 0]),
}


COLOR_TO_INDEX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
    'white': 6,
    'black': 7,
    'orange': 8,
}

LAYOUT_STR_TO_LAYOUT = {
    "asymm_advantages_6_9": asymm_advantages_6_9,
    "counter_circuit_6_9": counter_circuit_6_9,
    "forced_coord_6_9": forced_coord_6_9,
    "cramped_room_6_9": cramped_room_6_9,
    "coord_ring_6_9": coord_ring_6_9,
    "coord_ring_5_5": coord_ring_5_5,
    "forced_coord_5_5": forced_coord_5_5,
    "cramped_room_5_5": cramped_room_5_5,
}


OBJECT_INDEX_TO_VEC = jnp.array([
    jnp.array([OBJECT_TO_INDEX['unseen'], 0, 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0],
              dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['onion'],
              COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['onion_pile'],
              COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['plate'],
              COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['plate_pile'],
              COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0],
              dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX['black'], 0],
              dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX["white"], 0],
              dtype=jnp.uint8),
    jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], 0],
              dtype=jnp.uint8),  					# Default color and direction
])


# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
    # Pointing right (positive X)
    # (1, 0), # right
    # (0, 1), # down
    # (-1, 0), # left
    # (0, -1), # up
    (0, -1),  # NORTH
    (0, 1),  # SOUTH
    (1, 0),  # EAST
    (-1, 0),  # WEST
], dtype=jnp.int8)


def _obtain_from_layout(key, layout, h, w, random_reset, num_agents):
    all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)
    occupied_mask = layout.get("wall_idx")
    # occupied_mask = jnp.zeros_like(all_pos)
    # occupied_mask = occupied_mask.at[wall_idx].set(1)
    wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

    # Reset agent position + dir
    key, subkey = jax.random.split(key)
    agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                  p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.uint8), replace=False)
    # agent_idx = jnp.zeros_like(occupied_mask).at[agent_idx].set(1)

    # Replace with fixed layout if applicable. Also randomize if agent position not provided
    # agent_idx = random_reset*agent_idx + \ # (1-random_reset)*
    agent_idx = layout.get("agent_idx", agent_idx)
    agent_pos = jnp.array([agent_idx % w, agent_idx // w],
                          dtype=jnp.uint32).transpose()  # dim = n_agents x 2
    # agent_pos = agent_idx.reshape(h,w)
    occupied_mask = occupied_mask.at[agent_idx].set(1)

    key, subkey = jax.random.split(key)
    agent_dir_idx = jax.random.choice(subkey, jnp.arange(
        len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
    agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()  # dim = n_agents x 2

    empty_table_mask = jnp.zeros_like(all_pos)
    empty_table_mask = jnp.array(layout.get("empty_table_idx")).reshape(h, w)

    goal_idx = layout.get("goal_idx")
    goal_pos = goal_idx.reshape(h, w)
    empty_table_mask = empty_table_mask.at[goal_idx].set(0)

    onion_pile_idx = layout.get("onion_pile_idx")
    onion_pile_pos = onion_pile_idx.reshape(h, w)
    empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

    plate_pile_idx = layout.get("plate_pile_idx")
    plate_pile_pos = plate_pile_idx.reshape(h, w)
    empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

    pot_idx = layout.get("pot_idx")
    pot_pos = pot_idx.reshape(h, w)
    empty_table_mask = empty_table_mask.at[pot_idx].set(0)

    key, subkey = jax.random.split(key)
    pot_status = pot_idx * \
        jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24, dtype=jnp.uint8)
    pot_status = pot_status * random_reset + \
        (1-random_reset) * jnp.ones((pot_idx.shape[0]), dtype=jnp.uint8) * 23
    return wall_map, goal_pos, agent_pos, agent_dir, agent_dir_idx, plate_pile_pos, onion_pile_pos, pot_pos, pot_status


class Overcooked(environment.Environment):
    """Overcooked Procedural Multi-Agent"""

    def __init__(
        self,
        height: int,
        width: int,
        random_reset: bool = False,
        n_walls=25,
        agent_view_size=5,
        replace_wall_pos=False,
        max_steps=400,
        normalize_obs=False,
        sample_n_walls=False,
        fix_to_single_layout=None,
        dense_obs=False,
        singleton_seed=-1
    ):
        # Sets self.num_agents to 2
        super().__init__()

        self.num_agents = 2
        self.default_shaped_reward_coeff = 0.0
        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        # The idea is that we never create levels biger that this for zero padding.
        self.width = width
        self.height = height
        self.num_features = 62  # Akin to the original Overcooked-AI

        # Hard coded. Only affects map padding -- not observations.
        self.agent_view_size = 5
        self.agents = ["agent_0", "agent_1"]
        # Fixes Resets to this layout instead to a random one.
        # Mostly used for debugging.
        # Example: "asymm_advantages_6_9" -> asymm_advantages_6_9
        self.fix_to_single_layout = fix_to_single_layout
        self.dense_obs = dense_obs

        # Define the observation function
        if dense_obs:  # (62,)
            self.get_obs = self.get_obs_dense
            self.obs_shape = (self.num_features,)
        else:  # (h, w, 26,)
            self.get_obs = self.get_obs_sparse
            self.obs_shape = (self.width, self.height, 26)

        self.action_set = jnp.array([
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
            Actions.interact,
        ])

        self.random_reset = random_reset
        self.max_steps = max_steps

        self.params = EnvParams(
            height=height,
            width=width,
            n_walls=n_walls,
            agent_view_size=agent_view_size,
            replace_wall_pos=replace_wall_pos and not sample_n_walls,
            max_steps=max_steps,
            normalize_obs=normalize_obs,
            sample_n_walls=sample_n_walls,
            singleton_seed=-1,
            max_episode_steps=max_steps,
        )

    def step_env(
            self,
            key: chex.PRNGKey,
            state: EnvState,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array(
            [actions["agent_0"], actions["agent_1"]]))

        state, reward, shaped_reward_alice, shaped_reward_bob = self.step_agents(
            key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {
            "agent_0": reward,
            "agent_1": reward
        }
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {
                "sparse_reward": jnp.array([reward, reward]),
                "shaped_reward": jnp.array([shaped_reward_alice, shaped_reward_bob]),
            },
        )

    def sample_random_layout(self, key: chex.PRNGKey, h, w) -> Dict[str, chex.Array]:
        """Samples a random layout that might or might not be playable.
        """
        params = self.params

        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint8)

        key, walls_key, nwalls_key, goal_key, plate_pile_key, onion_pile_key, pot_key, agpos_key = jax.random.split(
            key, 8)
        wall_idx = jax.random.choice(
            walls_key, all_pos,
            shape=(params.n_walls,),
            replace=params.replace_wall_pos)

        if params.sample_n_walls:
            sampled_n_walls = jax.random.randint(
                nwalls_key, (), minval=0, maxval=params.n_walls)
            sample_wall_mask = jnp.arange(params.n_walls) < sampled_n_walls
            dummy_wall_idx = wall_idx.at[0].get().repeat(params.n_walls)
            wall_idx = jax.lax.select(
                sample_wall_mask,
                wall_idx,
                dummy_wall_idx
            )

        walls = jnp.zeros_like(all_pos, dtype=jnp.uint8)
        walls = walls.at[wall_idx].set(1)
        walls = walls.reshape(h, w)
        walls = walls.at[:, 0].set(1)
        walls = walls.at[0, :].set(1)
        walls = walls.at[:, -1].set(1)
        walls = walls.at[-1, :].set(1).reshape(-1)

        occupied_obj_mask = jnp.zeros_like(all_pos, dtype=jnp.uint8)
        wall_mask = occupied_obj_mask + walls

        # Do not want corners to have objects
        occupied_obj_mask = occupied_obj_mask.reshape(h, w)
        occupied_obj_mask = occupied_obj_mask.at[0, 0].set(1)
        occupied_obj_mask = occupied_obj_mask.at[-1, -1].set(1)
        occupied_obj_mask = occupied_obj_mask.at[0, -1].set(1)
        occupied_obj_mask = occupied_obj_mask.at[-1, 0].set(1)
        occupied_obj_mask = occupied_obj_mask.reshape(-1)

        def add_1_or_2_items(key, all_pos, wall_mask, occupied_obj_mask):
            # occupied_obj_mask is only objects on tables so we can do:
            possible_positions = wall_mask - occupied_obj_mask
            obj_mask = jnp.zeros_like(all_pos, dtype=jnp.uint8)
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            item_idx_1 = jax.random.choice(subkey1, all_pos, shape=(
                1,), p=(possible_positions.astype(jnp.bool_)).astype(jnp.uint8))

            and_2 = jax.random.bernoulli(subkey2, 0.5)

            item_idx_2 = jax.random.choice(subkey3, all_pos, shape=(
                1,), p=(possible_positions.astype(jnp.bool_)).astype(jnp.uint8))

            obj_mask = obj_mask.at[item_idx_1].set(1)

            update_2 = jnp.logical_or(
                obj_mask.at[item_idx_2].get(), and_2.astype(jnp.uint8))
            obj_mask = obj_mask.at[item_idx_2].set(update_2)
            return obj_mask

        goal_pos = add_1_or_2_items(
            goal_key, all_pos, wall_mask, occupied_obj_mask)
        occupied_obj_mask = occupied_obj_mask + goal_pos

        plate_pile_pos = add_1_or_2_items(
            plate_pile_key, all_pos, wall_mask, occupied_obj_mask)
        occupied_obj_mask = occupied_obj_mask + plate_pile_pos

        onion_pile_pos = add_1_or_2_items(
            onion_pile_key, all_pos, wall_mask, occupied_obj_mask)
        occupied_obj_mask = occupied_obj_mask + onion_pile_pos

        pot_pos = add_1_or_2_items(
            pot_key, all_pos, wall_mask, occupied_obj_mask)
        occupied_obj_mask = occupied_obj_mask + pot_pos

        agent_idx = jax.random.choice(agpos_key, all_pos, shape=(2,), replace=False, p=(
            ~wall_mask.astype(jnp.bool_)).astype(jnp.uint8))
        # occupied_mask = occupied_mask.at[agent_idx].set(2)

        layout = {
            "height": self.height,
            "width": self.width,
            "wall_idx": walls,
            "empty_table_idx": walls - occupied_obj_mask,
            "agent_idx": agent_idx,
            "goal_idx": goal_pos,
            "plate_pile_idx": plate_pile_pos,
            "onion_pile_idx": onion_pile_pos,
            "pot_idx": pot_pos
        }
        return layout

    def reset_env(  # NOTE: Has been renamed to fit minimax
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], EnvState]:
        """Reset environment state based on `self.random_reset`

        If True, everything is randomized, including agent inventories and positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by `self.layout`
        """
        # Whether to fully randomize the start state
        random_reset = self.random_reset

        h = self.height
        w = self.width
        num_agents = self.num_agents

        if self.fix_to_single_layout is None:
            layout = self.sample_random_layout(key, h, w)
        else:
            layout = layout_grid_to_onehot_dict(
                LAYOUT_STR_TO_LAYOUT[self.fix_to_single_layout])

        wall_map, goal_pos, agent_pos, agent_dir, agent_dir_idx, plate_pile_pos, onion_pile_pos, pot_pos, pot_status\
            = _obtain_from_layout(key, layout, h, w, random_reset, num_agents)

        onion_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        plate_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        dish_pos = jnp.zeros((h, w), dtype=jnp.uint8)

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )
        # Its to make padding static with respect to the jitted code later.
        # Its static since we compute it in advance now.
        padding = (maze_map.shape[0]-h) // 2

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(
            subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
            (1-random_reset) * \
            jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            onion_pile_pos=onion_pile_pos,
            bowl_pile_pos=plate_pile_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        self.padding = padding
        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs_dense(self, state: EnvState) -> Dict[str, chex.Array]:
        """
        Inspired by the original Overcooked-AI we also add a dense observation to the environment.
        We use this to built the OvercookedUED light challange as it is significantly less sparse then the original observation.

        From their doc (https://github.com/HumanCompatibleAI/overcooked_ai/blob/cff884ccf5709658ee4cd489e63367200b4c86d6/src/overcooked_ai_py/mdp/overcooked_mdp.py#L2579):
        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """
        agent_dir = state.agent_dir
        agent_inv = state.agent_inv
        maze_map = state.maze_map

        w = self.width
        h = self.height

        padding = 4
        maze_map = maze_map[padding:-padding, padding:-padding, :]

        def get_player_rep(player_idx: int, state):

            agent_pos = state.agent_pos[player_idx]

            # pi_orientation: length 4 one-hot-encoding of direction currently facing
            pi_orientation = jnp.zeros((4)).at[state.agent_dir_idx].set(1)

            # pi_obj: length 3 one-hot-encoding of object currently being held (all 0s if no object held)
            pi_obj = OBJECT_TO_INDEX["empty"] + (agent_inv[player_idx] == OBJECT_TO_INDEX["onion"]) * jnp.array([1, 0, 0], dtype=jnp.uint8)\
                + (agent_inv[player_idx] == OBJECT_TO_INDEX["plate"]) * jnp.array([0, 1, 0], dtype=jnp.uint8)\
                + (agent_inv[player_idx] == OBJECT_TO_INDEX["dish"]
                   ) * jnp.array([0, 0, 1], dtype=jnp.uint8)

            # pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
            fwd_pos_0 = agent_pos + DIR_TO_VEC[0]
            is_wall_0 = state.wall_map.at[fwd_pos_0[1], fwd_pos_0[0]].get()

            fwd_pos_1 = agent_pos + DIR_TO_VEC[1]
            is_wall_1 = state.wall_map.at[fwd_pos_1[1], fwd_pos_1[0]].get()

            fwd_pos_2 = agent_pos + DIR_TO_VEC[2]
            is_wall_2 = state.wall_map.at[fwd_pos_2[1], fwd_pos_2[0]].get()

            fwd_pos_3 = agent_pos + DIR_TO_VEC[3]
            is_wall_3 = state.wall_map.at[fwd_pos_3[1], fwd_pos_3[0]].get()

            pi_wall_j = jnp.array([is_wall_0, is_wall_1, is_wall_2, is_wall_3])

            # pi_closest_{onion|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
            def find_closest_between_masks(agent_pos, object_map, name):
                obj_idx = OBJECT_TO_INDEX[name]
                padded_pos = jnp.argwhere(
                    object_map.T == obj_idx, size=2,  # w*h,
                    fill_value=jnp.inf)
                dist = padded_pos-agent_pos
                abs_dist = jnp.abs(dist)
                manhatten = abs_dist.sum(-1)
                closest_idx = jnp.argmin(manhatten)
                clostest_obj_pos = padded_pos[closest_idx]
                dxdy_obj_ag_inf = dist[closest_idx]
                dxdy_obj_ag = jnp.nan_to_num(dxdy_obj_ag_inf, nan=0, posinf=0)
                return clostest_obj_pos.astype(jnp.uint8), dxdy_obj_ag

            object_map = maze_map[:, :, 0]
            pos_closest_pot, pi_closest_pot = find_closest_between_masks(
                agent_pos, object_map, "pot")
            _, pi_closest_onion = find_closest_between_masks(
                agent_pos, object_map, "onion")
            _, pi_closest_plate = find_closest_between_masks(
                agent_pos, object_map, "plate")
            _, pi_closest_dish = find_closest_between_masks(
                agent_pos, object_map, "dish")
            _, pi_closest_goal = find_closest_between_masks(
                agent_pos, object_map, "goal")
            # If it has something on it its type is not wall -> i.e. walls are always empty
            # empty_wall_map = (maze_map[:,:,0] == OBJECT_TO_INDEX["wall"]).astype(jnp.uint8)
            _, pi_closest_wall = find_closest_between_masks(
                agent_pos, object_map, "wall")

            # pi_cloest_soup_n_{onions}: int value for number of this ingredient in closest soup
            # Not apllicable: We only have 3 onion soups
            # pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
            # be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
            # pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
            # pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
            # pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
            # pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location
            closest_pot = maze_map.at[pos_closest_pot[1],
                                      pos_closest_pot[0]].get()

            # agent_obj = maze_map.at[agent_pos[1], agent_pos[0]].get()

            path_len = _graph_util.shortest_path_len(
                state.wall_map, agent_pos, pos_closest_pot)

            # pi_closest_pot_{j}_exists
            pi_closest_pot_exists = path_len > 0
            pi_closest_pot_is_empty = (
                closest_pot[2] == 23) * pi_closest_pot_exists
            pi_closest_pot_is_full = (
                jnp.logical_and(closest_pot[2] <= 20, closest_pot[2] > 0)) * pi_closest_pot_exists
            pi_closest_pot_is_cooking = (
                jnp.logical_and(closest_pot[2] <= 19, closest_pot[2] > 0)) * pi_closest_pot_exists
            pi_closest_pot_is_ready = (
                closest_pot[2] == 0) * pi_closest_pot_exists
            pi_closest_pot_num_onions = (
                (closest_pot[2] <= 20)*3 + (closest_pot[2] == 21)*2 + (closest_pot[2] == 22)*1) * pi_closest_pot_exists
            pi_closest_pot_cook_time = pi_closest_pot_is_cooking * \
                closest_pot[2]

            return jnp.hstack([
                pi_orientation, pi_obj, pi_wall_j, pi_closest_onion, pi_closest_plate, pi_closest_dish,
                pi_closest_goal, pi_closest_wall, pi_closest_pot_exists, pi_closest_pot_is_empty, pi_closest_pot_is_full,
                pi_closest_pot_is_cooking, pi_closest_pot_is_ready, pi_closest_pot_num_onions, pi_closest_pot_cook_time, pi_closest_pot
            ])

        agent_vec_0 = get_player_rep(0, state)
        agent_vec_1 = get_player_rep(1, state)

        obs = {
            'agent_0': jnp.hstack([agent_vec_0, agent_vec_1, state.agent_pos[0, 1], state.agent_pos[0, 0]]),
            'agent_1': jnp.hstack([agent_vec_1, agent_vec_0, state.agent_pos[1, 1], state.agent_pos[1, 0]])
        }
        return obs

    def get_obs_sparse(self, state: EnvState) -> Dict[str, chex.Array]:
        """Return a full observation, of size(height x width x n_layers), where n_layers = 26.
        Layers are of shape(height x width) and are binary(0/1) except where indicated otherwise.
        The obs is very sparse(most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions:
        0. position of agent i(1 at agent loc, 0 otherwise)
        1. position of agent(1-i)

        Agent orientations:
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3(layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions(1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations(table)
        12. onion pile locations
        13. tomato pile locations(tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations(goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot(0, 1, 2, 3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup(0, 3) for elements corresponding to either a cooking/done pot or to a soup(dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup(dish). O otherwise.

        Variable env layers(binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """
        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        # NOTE: Original code here was: padding = (state.maze_map.shape[0]-height) // 2
        padding = 4
        # padding = state.padding # Must be somehow static

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(
            maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(
            maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding,
                                    padding: -padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (
            pot_status >= POT_FULL_STATUS)    # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
            * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc   # 0/3, as long as cooking or done
        pot_cooking_time_layer = pot_status * \
            (pot_status < POT_FULL_STATUS)                           # Timer: 19 to 0
        # Ready soups, plated or not
        soup_ready_layer = pot_loc_layer * \
            (pot_status == POT_READY_STATUS) + soup_loc
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * \
            ((self.max_steps - state.time) < URGENCY_CUTOFF)

        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0,
                                               state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1,
                                               state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(
            state.agent_inv, (1, 2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers, 0),
                             agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer
        + (jnp.sum(agent_inv_items, 0) ==
           OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers, 0)
        onions_in_soup_layer = onions_in_soup_layer \
            + (jnp.sum(agent_inv_items, 0) ==
               OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers, 0)

        env_layers = [
            # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(
                maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            # tomato pile
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),
            jnp.array(
                maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            # 15
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            # tomatoes in pot
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            # tomatoes in soup
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),
            jnp.array(pot_cooking_time_layer,
                      dtype=jnp.uint8),                     # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            # tomatoes
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),
            urgency_layer,                                                          # 25
        ]

        # Agent related layers
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layer_idx = state.agent_dir_idx+jnp.array([0, 4])
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx, :, :].set(
            agent_pos_layers)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)

        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(
            agent_pos_layers[1]).at[1].set(agent_pos_layers[0])
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:]).at[6:10].set(
            agent_direction_layers[0:4])
        bob_obs = bob_obs.at[10:].set(jnp.stack(env_layers))

        # NOTE: Changed, was not inline with self.obs_shape: [self.width, self.height, 26]
        alice_obs = jnp.transpose(alice_obs, (2, 1, 0))
        bob_obs = jnp.transpose(bob_obs, (2, 1, 0))
        return {"agent_0": alice_obs, "agent_1": bob_obs}

    def step_agents(
            self, key: chex.PRNGKey, state: EnvState, action: chex.Array
    ) -> Tuple[EnvState, float]:

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(
            action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(
            is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)]
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            def goal_collision(pos, goal): return jnp.logical_and(
                pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(
                None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(
            0, None, None))(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(
            fwd_pos_has_wall, fwd_pos_has_goal).reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced)
                   * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],                     # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            # collision and Alice bounced
            state.agent_pos[1],
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[1],
            bob_pos
        )

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward, alice_shaped_reward = self.process_interact(
            maze_map, state, fwd_pos[0], state.agent_inv[0], state.agent_inv[1])
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact,
                                  candidate_maze_map,
                                  maze_map)
        alice_inv = jax.lax.select(alice_interact,
                                   alice_inv,
                                   state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.)
        alice_shaped_reward = jax.lax.select(
            alice_interact, alice_shaped_reward, 0.)

        candidate_maze_map, bob_inv, bob_reward, bob_shaped_reward = self.process_interact(
            maze_map, state, fwd_pos[1], state.agent_inv[1], state.agent_inv[0])
        maze_map = jax.lax.select(bob_interact,
                                  candidate_maze_map,
                                  maze_map)
        bob_inv = jax.lax.select(bob_interact,
                                 bob_inv,
                                 state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.)
        bob_shaped_reward = jax.lax.select(bob_interact, bob_shaped_reward, 0.)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'] +
                              agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(
            agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        # height = self.obs_shape[1]
        padding = 4  # (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev,
                               padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y,
                               padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(maze_map, pot_pos):
            pot_pos_padded = jnp.zeros(
                (maze_map.shape[0], maze_map.shape[1]), dtype=jnp.uint8
            )
            pot_pos_padded = pot_pos_padded.at[
                padding:-padding, padding:-padding].set(pot_pos)
            is_cooking = jnp.array(
                maze_map[:, :, -1] * pot_pos_padded <= POT_FULL_STATUS, dtype=jnp.uint8) * pot_pos_padded
            not_done = jnp.array(
                maze_map[:, :, -1] * pot_pos_padded > POT_READY_STATUS, dtype=jnp.uint8) * pot_pos_padded
            pot_status_is_cooking_not_done = is_cooking * \
                not_done * (maze_map[:, :, -1] - 1) * pot_pos_padded
            pot_status_is_not_cooking = jnp.logical_not(
                is_cooking) * (maze_map[:, :, -1]) * pot_pos_padded  # defaults to zero if done pot_status
            pot_status = pot_status_is_cooking_not_done + pot_status_is_not_cooking

            pot_status_map = pot_pos_padded * pot_status + \
                jnp.logical_not(pot_pos_padded) * maze_map[:, :, -1]
            pot_status_map = jnp.concatenate(
                (jnp.zeros((*pot_status_map.shape, 2), dtype=jnp.uint8), pot_status_map[:, :, jnp.newaxis]), axis=-1)

            pot_pos_3 = jnp.concatenate(
                (jnp.zeros((pot_status_map.shape[0], pot_status_map.shape[1], 2), dtype=jnp.uint8), pot_pos_padded[:, :, jnp.newaxis]), axis=-1)

            maze_map = maze_map * (1-pot_pos_3) + pot_status_map * pot_pos_3

            return maze_map  # pot.at[-1].set(pot_status)

        maze_map = _cook_pots(maze_map, state.pot_pos)

        reward = alice_reward + bob_reward
        # shaped_reward = alice_shaped_reward + bob_shaped_reward

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            reward,
            alice_shaped_reward,
            bob_shaped_reward,
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            state: EnvState,
            fwd_pos: chex.Array,
            inventory: chex.Array,
            other_inventory: chex.Array):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

        wall_map = state.wall_map
        height = self.height  # self.obs_shape[1]
        # padding = (maze_map.shape[0] - height) // 2
        padding = 4

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding +
                                           fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["plate_pile"], object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(
            object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(
                object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"]
        )
        # Whether the object in front is counter space that the agent can drop on.
        is_table = jnp.logical_and(
            wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["wall"], object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * \
            holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * \
            (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # Update pot status and object in inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        successful_pickup = is_table * ~table_is_empty * inv_is_empty * \
            jnp.logical_or(object_is_pile, object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(
            ~successful_pickup, ~successful_drop), ~successful_delivery)

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # Apply inventory update
        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1],
                               padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float)*DELIVERY_REWARD

        no_plate_on_counter = (
            (maze_map[padding:-padding, padding:-padding, 0] * wall_map) == OBJECT_TO_INDEX["plate"]).sum() == 0
        num_pots = state.pot_pos.sum()
        #  (maze_map[padding:-padding, padding:-padding, -1].at[state.pot_pos].get() <= POT_FULL_STATUS).sum()
        num_pots_cooking = (
            (maze_map[padding:-padding, padding:-padding, -1] <= POT_FULL_STATUS) * state.pot_pos).sum()
        #  (maze_map[padding:-padding, padding:-padding, -1].at[state.pot_pos].get()  > POT_FULL_STATUS).sum()
        num_pots_not_started = (
            (maze_map[padding:-padding, padding:-padding, -1] > POT_FULL_STATUS) * state.pot_pos).sum()
        num_pots_ready = num_pots - num_pots_cooking - num_pots_not_started
        pot_left_over_for_plate = (num_pots_cooking + num_pots_ready -
                                   1 * (other_inventory == OBJECT_TO_INDEX["dish"])) > 0
        # As in orignal work: adding onion 3, getting a bowl while cooking 5, pickung up a soup 5
        shaped_reward_c1 = (new_object_in_inv == OBJECT_TO_INDEX["empty"]) * (
            object_in_inv == OBJECT_TO_INDEX["onion"]) * case_1 * 3.0
        shaped_reward_c2 = (new_object_in_inv == OBJECT_TO_INDEX["plate"]) * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * \
            successful_pickup * no_plate_on_counter * pot_left_over_for_plate * 5.0
        shaped_reward_c3 = (new_object_in_inv == OBJECT_TO_INDEX["dish"]) * (
            object_in_inv == OBJECT_TO_INDEX["plate"]) * case_2 * 5.0

        # jax.debug.print("no_plate {a}: {s}", a=no_plate_on_counter, s=shaped_reward_c2)
        shaped_reward = shaped_reward_c1 + shaped_reward_c2 + shaped_reward_c3
        return maze_map, inventory, reward, shaped_reward

    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 20  # More than one soup delivered

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return spaces.Discrete(
            len(self.action_set),
            dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)

    def max_episode_steps(self) -> int:
        return self.params.max_episode_steps

    def set_env_instance(
            self,
            encoding: EnvInstance):
        """
        Instance is encoded as a PyTree containing the following fields:
        agent_pos, agent_dir, goal_pos, wall_map
        """
        params = self.params
        agent_pos = encoding.agent_pos
        agent_dir_idx = encoding.agent_dir_idx
        h, w = encoding.wall_map.shape
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()
        goal_pos = encoding.goal_pos
        wall_map = encoding.wall_map
        agent_inv = encoding.agent_inv
        pot_pos = encoding.pot_pos

        onion_pile_pos = encoding.onion_pile_pos
        plate_pile_pos = encoding.plate_pile_pos

        onion_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        plate_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        dish_pos = jnp.zeros((h, w), dtype=jnp.uint8)

        pot_status = jnp.ones(
            (encoding.wall_map.reshape(-1).shape), dtype=jnp.uint8) * 23

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=2,
            agent_view_size=5)

        state = EnvState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            goal_pos=goal_pos,
            wall_map=wall_map,
            maze_map=maze_map,
            bowl_pile_pos=plate_pile_pos,
            onion_pile_pos=onion_pile_pos,
            agent_inv=agent_inv,
            pot_pos=pot_pos,
            time=0,
            terminal=False
        )

        return self.get_obs(state), state

    def get_env_metrics(self, state: EnvState) -> dict:
        n_walls = state.wall_map.sum()
        return dict(
            n_walls=n_walls,
        )

    def state_space(self) -> spaces.Dict:
        """EnvState space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps

    def get_monitored_metrics(self):
        return ('reward', 'shaped_reward', 'shaped_reward_scaled_by_shaped_reward_coeff', 'reward_p_shaped_reward_scaled')

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register(env_id='Overcooked', entry_point=module_path + ':Overcooked')

if __name__ == '__main__':
    from minimax.envs.wrappers import MonitorReturnWrapper

    render = False
    n_envs = 1

    kwargs = dict(
        # max_episode_steps=400,
        height=6,
        width=9,
        n_walls=15,
        agent_view_size=5,
        fix_to_single_layout="coord_ring_6_9"
    )
    env = MonitorReturnWrapper(Overcooked(**kwargs))
    params = env.params
    extra = env.reset_extra()

    jit_reset_env = env.reset
    jit_step_env = env.step

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state, extra = jit_reset_env(subkey)

    all_sps = []

    import time
    for ac in [0, 0, 5, 0, 0]:  # [1, 1, 3, 1, 5]:
        key, subkey = jax.random.split(key)
        # vrngs = jax.random.split(subkey)
        start = time.time()
        jax.debug.print('obs:\n{a}', a=(obs['agent_0'][:, :, 0]
                        * 1 + obs['agent_0'][:, :, 1]*2+obs['agent_0'][:, :, 11]*3).T)
        obs, state, reward, done, info, extra = jit_step_env(
            subkey,
            state,
            action={
                'agent_0': ac,
                'agent_1': ac
            },
            extra=extra
        )
        jax.debug.print("reward r {r} {ir} {isr}", r=reward,
                        ir=info["sparse_reward"], isr=info["shaped_reward"])

        state = state.replace(agent_inv=jnp.array(
            [OBJECT_TO_INDEX['onion'], OBJECT_TO_INDEX['onion']]))

        obs['agent_0'].block_until_ready()
        end = time.time()
        # print(f"sps: {1/(end-start) * n_envs}")
        # print('return:', info['return'])
        all_sps.append(1/(end-start) * n_envs)

    print('mean sps:', np.mean(all_sps))
    print('std sps:', np.std(all_sps))
