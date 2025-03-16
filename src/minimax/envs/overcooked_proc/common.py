# Edited from JaxMarl: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked

from flax import struct
import chex
import numpy as np
import jax.numpy as jnp
import jax


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


@struct.dataclass
class EnvInstance:
    agent_pos: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    onion_pile_pos: chex.Array
    plate_pile_pos: chex.Array
    wall_map: chex.Array


def make_overcooked_map(
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
        agent_view_size=5):
    # Expand maze map to H x W x C
    empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
    wall = jnp.array(
        [OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
    maze_map = jnp.array(jnp.expand_dims(wall_map, -1), dtype=jnp.uint8)
    maze_map = jnp.where(maze_map > 0, wall, empty)

    # Add agents
    def _get_agent_updates(agent_dir_idx, agent_pos, agent_idx):
        agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'] +
                          agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
        agent_x, agent_y = agent_pos
        return agent_x, agent_y, agent

    agent_x_vec, agent_y_vec, agent_vec = jax.vmap(_get_agent_updates, in_axes=(
        0, 0, 0))(agent_dir_idx, agent_pos, jnp.arange(num_agents))
    maze_map = maze_map.at[agent_y_vec, agent_x_vec, :].set(agent_vec)

    # Add goals
    goal = jnp.array(
        [OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)

    def set_based_on_position_mask(maze_map, pos_mask, obj):
        pos_expanded = jnp.repeat(
            jnp.expand_dims(pos_mask, axis=-1), 3, axis=-1)
        obj_maze_map = pos_expanded * jnp.tile(obj, (*pos_mask.shape, 1))
        maze_map = maze_map * \
            jnp.logical_not(pos_expanded) + obj_maze_map * pos_expanded
        return maze_map

    maze_map = set_based_on_position_mask(maze_map, goal_pos, goal)

    # Add onions
    onion_pile = jnp.array(
        [OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8)
    maze_map = set_based_on_position_mask(maze_map, onion_pile_pos, onion_pile)

    # Add plates
    plate_pile = jnp.array(
        [OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
    maze_map = set_based_on_position_mask(maze_map, plate_pile_pos, plate_pile)

    pot_obj = jnp.array(
        [OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX["black"], 0], dtype=jnp.uint8)
    pot_status = pot_status.reshape(pot_pos.shape)
    pot_status = jnp.concatenate((jnp.zeros(
        (*pot_status.shape, 2), dtype=jnp.uint8), pot_status[:, :, jnp.newaxis]), axis=-1)
    pos_expanded = jnp.repeat(jnp.expand_dims(pot_pos, axis=-1), 3, axis=-1)
    obj_maze_map = pos_expanded * \
        jnp.tile(pot_obj, (*pot_pos.shape, 1)) + pot_status
    maze_map = maze_map * \
        jnp.logical_not(pos_expanded) + obj_maze_map * pos_expanded

    onion = jnp.array(
        [OBJECT_TO_INDEX['onion'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8)
    maze_map = set_based_on_position_mask(maze_map, onion_pos, onion)

    plate = jnp.array(
        [OBJECT_TO_INDEX['plate'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
    maze_map = set_based_on_position_mask(maze_map, plate_pos, plate)

    dish = jnp.array(
        [OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8)
    maze_map = set_based_on_position_mask(maze_map, dish_pos, dish)

    # Add observation padding
    if pad_obs:
        padding = agent_view_size-1
    else:
        padding = 1

    maze_map_padded = jnp.tile(wall.reshape(
        (1, 1, *empty.shape)), (maze_map.shape[0]+2*padding, maze_map.shape[1]+2*padding, 1))
    maze_map_padded = maze_map_padded.at[
        padding:-padding, padding:-padding, :].set(maze_map)

    # Add surrounding walls
    wall_start = padding-1  # start index for walls
    wall_end_y = maze_map_padded.shape[0] - wall_start - 1
    wall_end_x = maze_map_padded.shape[1] - wall_start - 1
    maze_map_padded = maze_map_padded.at[wall_start,
                                         wall_start:wall_end_x+1, :].set(wall)  # top
    maze_map_padded = maze_map_padded.at[wall_end_y,
                                         wall_start:wall_end_x+1, :].set(wall)  # bottom
    # left
    maze_map_padded = maze_map_padded.at[wall_start:wall_end_y +
                                         1, wall_start, :].set(wall)
    # right
    maze_map_padded = maze_map_padded.at[wall_start:wall_end_y +
                                         1, wall_end_x, :].set(wall)

    return maze_map_padded
