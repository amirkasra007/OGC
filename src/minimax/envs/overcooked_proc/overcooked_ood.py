"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import chex

from flax.core.frozen_dict import FrozenDict

from minimax.envs.registration import register
from minimax.envs.overcooked_proc.layouts import layout_grid_to_onehot_dict
from .common import (
    OBJECT_TO_INDEX,
    make_overcooked_map,
)
from .overcooked import (
    Overcooked,
    EnvParams,
    EnvState,
    _obtain_from_layout,

)

cramped_room = {
    "height": 4,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 9,
                           10, 14,
                           15, 16, 17, 18, 19]),
    "agent_idx": jnp.array([6, 8]),
    "goal_idx": jnp.array([18]),
    "plate_pile_idx": jnp.array([16]),
    "onion_pile_idx": jnp.array([5, 9]),
    "pot_idx": jnp.array([2])
}
asymm_advantages = {
    "height": 5,
    "width": 9,
    "wall_idx": jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                           9, 11, 12, 13, 14, 15, 17,
                           18, 22, 26,
                           27, 31, 35,
                           36, 37, 38, 39, 40, 41, 42, 43, 44]),
    "agent_idx": jnp.array([29, 32]),
    "goal_idx": jnp.array([12, 17]),
    "plate_pile_idx": jnp.array([39, 41]),
    "onion_pile_idx": jnp.array([9, 14]),
    "pot_idx": jnp.array([22, 31])
}
coord_ring = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 9,
                           10, 12, 14,
                           15, 19,
                           20, 21, 22, 23, 24]),
    "agent_idx": jnp.array([7, 11]),
    "goal_idx": jnp.array([22]),
    "plate_pile_idx": jnp.array([10]),
    "onion_pile_idx": jnp.array([15, 21]),
    "pot_idx": jnp.array([3, 9])
}
forced_coord = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4,
                           5, 7, 9,
                           10, 12, 14,
                           15, 17, 19,
                           20, 21, 22, 23, 24]),
    "agent_idx": jnp.array([11, 8]),
    "goal_idx": jnp.array([23]),
    "onion_pile_idx": jnp.array([5, 10]),
    "plate_pile_idx": jnp.array([15]),
    "pot_idx": jnp.array([3, 9])
}

# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

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


# ======== Singleton mazes ========
class OvercookedSingleton(Overcooked):
    def __init__(
        self,
        grid,
        agent_view_size=5,
        replace_wall_pos=False,
        max_steps=400,
        normalize_obs=False,
        sample_n_walls=False,
        singleton_seed=-1
    ):
        height = grid["height"]
        width = grid["width"]
        super().__init__(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            replace_wall_pos=replace_wall_pos and not sample_n_walls,
            max_steps=max_steps,
            normalize_obs=normalize_obs,
            sample_n_walls=sample_n_walls,
            singleton_seed=singleton_seed,
        )

        self.params = EnvParams(
            height=height,
            width=width,
            agent_view_size=agent_view_size,
            normalize_obs=normalize_obs,
            max_steps=max_steps,
            singleton_seed=singleton_seed,
        )

        h = self.height
        w = self.width

        # NOTE: that since the layout is fixed, the random_reset is set to False
        # and this is why jax.random.PRNGKey(0) is used too (not needed if no random_reset).
        wall_map, goal_pos, agent_pos, agent_dir, agent_dir_idx, plate_pile_pos, onion_pile_pos, pot_pos, pot_status\
            = _obtain_from_layout(jax.random.PRNGKey(0), grid, h, w, random_reset=False, num_agents=2)

        onion_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        plate_pos = jnp.zeros((h, w), dtype=jnp.uint8)
        dish_pos = jnp.zeros((h, w), dtype=jnp.uint8)

        agent_inv = jnp.array(
            [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        self.overcooked_map = make_overcooked_map(
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
            agent_view_size=self.agent_view_size)

        self.agent_pos = agent_pos
        self.agent_dir = agent_dir
        self.agent_dir_idx = agent_dir_idx
        self.agent_inv = agent_inv
        self.goal_pos = goal_pos
        self.pot_pos = pot_pos
        self.bowl_pile_pos = plate_pile_pos
        self.onion_pile_pos = onion_pile_pos
        self.wall_map = wall_map

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def reset_env(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, EnvState]:

        state = EnvState(
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            agent_dir_idx=self.agent_dir_idx,
            agent_inv=self.agent_inv,
            goal_pos=self.goal_pos,
            pot_pos=self.pot_pos,
            wall_map=self.wall_map.astype(jnp.bool_),
            maze_map=self.overcooked_map,
            bowl_pile_pos=self.bowl_pile_pos,
            onion_pile_pos=self.onion_pile_pos,
            time=0,
            terminal=False,
        )

        return self.get_obs(state), state


# ======== Specific mazes ========
class CoordRing6_9(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "coord_ring_6_9"

        grid = layout_grid_to_onehot_dict(coord_ring_6_9)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class ForcedCoord6_9(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "forced_coord_6_9"

        grid = layout_grid_to_onehot_dict(forced_coord_6_9)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class CounterCircuit6_9(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "counter_circuit_6_9"

        grid = layout_grid_to_onehot_dict(counter_circuit_6_9)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class AsymmAdvantages6_9(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "asymm_advantages_6_9"

        grid = layout_grid_to_onehot_dict(asymm_advantages_6_9)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class CrampedRoom6_9(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "cramped_room_6_9"

        grid = layout_grid_to_onehot_dict(cramped_room_6_9)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class CoordRing5_5(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "coord_ring_5_5"

        grid = layout_grid_to_onehot_dict(coord_ring_5_5)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class ForcedCoord5_5(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "forced_coord_5_5"

        grid = layout_grid_to_onehot_dict(forced_coord_5_5)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


class CrampedRoom5_5(OvercookedSingleton):
    def __init__(
            self,
            normalize_obs=False):
        self.layout_name = "cramped_room_5_5"

        grid = layout_grid_to_onehot_dict(cramped_room_5_5)

        super().__init__(
            grid=grid,
            normalize_obs=normalize_obs,
        )


# ======== Registration ========
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

# register(env_id='Overcooked', entry_point=module_path + ':')
register(env_id='Overcooked-CoordRing6_9',
         entry_point=module_path + ':CoordRing6_9')
register(env_id='Overcooked-ForcedCoord6_9',
         entry_point=module_path + ':ForcedCoord6_9')
register(env_id='Overcooked-CounterCircuit6_9',
         entry_point=module_path + ':CounterCircuit6_9')
register(env_id='Overcooked-AsymmAdvantages6_9',
         entry_point=module_path + ':AsymmAdvantages6_9')
register(env_id='Overcooked-CrampedRoom6_9',
         entry_point=module_path + ':CrampedRoom6_9')

register(env_id='Overcooked-CoordRing5_5',
         entry_point=module_path + ':CoordRing5_5')
register(env_id='Overcooked-ForcedCoord5_5',
         entry_point=module_path + ':ForcedCoord5_5')
register(env_id='Overcooked-CrampedRoom5_5',
         entry_point=module_path + ':CrampedRoom5_5')
