"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp

from .common import make_overcooked_map
from minimax.envs.registration import register_mutator


class Mutations(IntEnum):
    # Turn left, turn right, move forward
    NO_OP = 0
    FLIP_WALL = 1
    MOVE_GOAL = 2
    MOVE_POT = 3
    MOVE_ONION_PILE = 4
    MOVE_PLATE_PILE = 5


def add_1_or_2_items(key, all_pos, legal_pick_mask):
    obj_mask = jnp.zeros_like(all_pos, dtype=jnp.uint8)
    key, subkey = jax.random.split(key)
    item_idx_1 = jax.random.choice(subkey, all_pos, shape=(
        1,), p=(legal_pick_mask.astype(jnp.bool_)).astype(jnp.uint8))

    key, subkey = jax.random.split(key)
    and_2 = jax.random.bernoulli(subkey, 0.5)

    key, subkey = jax.random.split(key)
    item_idx_2 = jax.random.choice(subkey, all_pos, shape=(
        1,), p=(legal_pick_mask.astype(jnp.bool_)).astype(jnp.uint8))

    obj_mask = obj_mask.at[item_idx_1].set(1)

    update_2 = jnp.logical_or(
        obj_mask.at[item_idx_2].get(), and_2.astype(jnp.uint8))
    obj_mask = obj_mask.at[item_idx_2].set(update_2)
    return obj_mask


def flip_wall(rng, state):
    wall_map = state.wall_map
    h, w = wall_map.shape
    wall_mask = jnp.ones((h*w,), dtype=jnp.bool_)

    goal_idx = state.goal_pos.flatten()
    agent_idx = state.agent_pos.flatten()
    pot_pos_idx = state.pot_pos.flatten()
    onion_pile_pos_idx = state.onion_pile_pos.flatten()
    plate_pile_pos_idx = state.bowl_pile_pos.flatten()

    # Do not flip wall below an object or agent
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[agent_idx].set(False)
    wall_mask = wall_mask.at[pot_pos_idx].set(False)
    wall_mask = wall_mask.at[onion_pile_pos_idx].set(False)
    wall_mask = wall_mask.at[plate_pile_pos_idx].set(False)

    # Never allowed to flip a edge in overcooked
    wall_mask = wall_mask.reshape(h, w)
    wall_mask = wall_mask.at[:, 0].set(False)
    wall_mask = wall_mask.at[:, -1].set(False)
    wall_mask = wall_mask.at[0, :].set(False)
    wall_mask = wall_mask.at[-1, :].set(False)
    wall_mask = wall_mask.flatten()

    flip_idx = jax.random.choice(rng, np.arange(h*w), shape=(), p=wall_mask)

    wall_map = wall_map.flatten()
    flip_val = ~wall_map.at[flip_idx].get()

    wall_map = wall_map.at[flip_idx].set(flip_val)
    next_wall_map = wall_map.reshape(state.wall_map.shape)
    return state.replace(wall_map=next_wall_map)


def move_goal(rng, state):
    wall_map = state.wall_map
    h, w = wall_map.shape
    wall_mask = wall_map.flatten()

    onion_pile_pos_idx = state.onion_pile_pos.flatten()
    bowl_pile_pos_idx = state.bowl_pile_pos.flatten()
    goal_idx = state.goal_pos.flatten()
    pot_idx = state.pot_pos.flatten()

    # No previous position and other objects
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[pot_idx].set(False)
    wall_mask = wall_mask.at[bowl_pile_pos_idx].set(False)
    wall_mask = wall_mask.at[onion_pile_pos_idx].set(False)

    # Move around the wall
    all_pos = jnp.zeros((h*w,), dtype=jnp.uint8)
    next_goal_pos = add_1_or_2_items(rng, all_pos, wall_mask)
    return state.replace(goal_pos=next_goal_pos.reshape(state.goal_pos.shape))


def move_pot(rng, state):
    wall_map = state.wall_map
    h, w = wall_map.shape
    wall_mask = wall_map.flatten()

    onion_pile_pos_idx = state.onion_pile_pos.flatten()
    bowl_pile_pos_idx = state.bowl_pile_pos.flatten()
    goal_idx = state.goal_pos.flatten()
    pot_idx = state.pot_pos.flatten()

    # No previous position and other objects
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[pot_idx].set(False)
    wall_mask = wall_mask.at[bowl_pile_pos_idx].set(False)
    wall_mask = wall_mask.at[onion_pile_pos_idx].set(False)

    # Move around the wall
    all_pos = jnp.zeros((h*w,), dtype=jnp.uint8)
    next_pot_pos = add_1_or_2_items(rng, all_pos, wall_mask)
    return state.replace(pot_pos=next_pot_pos.reshape(state.pot_pos.shape))


def move_onion_pile(rng, state):
    wall_map = state.wall_map
    h, w = wall_map.shape
    wall_mask = wall_map.flatten()

    onion_pile_pos_idx = state.onion_pile_pos.flatten()
    bowl_pile_pos_idx = state.bowl_pile_pos.flatten()
    goal_idx = state.goal_pos.flatten()
    pot_idx = state.pot_pos.flatten()

    # No previous position and other objects
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[pot_idx].set(False)
    wall_mask = wall_mask.at[bowl_pile_pos_idx].set(False)
    wall_mask = wall_mask.at[onion_pile_pos_idx].set(False)

    # Move around the wall
    all_pos = jnp.zeros((h*w,), dtype=jnp.uint8)
    next_onion_pile_pos = add_1_or_2_items(rng, all_pos, wall_mask)
    return state.replace(onion_pile_pos=next_onion_pile_pos.reshape(state.onion_pile_pos.shape))


def move_bowl_pile(rng, state):
    wall_map = state.wall_map
    h, w = wall_map.shape
    wall_mask = wall_map.flatten()

    onion_pile_pos_idx = state.onion_pile_pos.flatten()
    bowl_pile_pos_idx = state.bowl_pile_pos.flatten()
    goal_idx = state.goal_pos.flatten()
    pot_idx = state.pot_pos.flatten()

    # No previous position and other objects
    wall_mask = wall_mask.at[goal_idx].set(False)
    wall_mask = wall_mask.at[pot_idx].set(False)
    wall_mask = wall_mask.at[bowl_pile_pos_idx].set(False)
    wall_mask = wall_mask.at[onion_pile_pos_idx].set(False)

    # Move around the wall
    all_pos = jnp.zeros((h*w,), dtype=jnp.uint8)
    next_plate_pile_pos = add_1_or_2_items(rng, all_pos, wall_mask)
    return state.replace(bowl_pile_pos=next_plate_pile_pos.reshape(state.bowl_pile_pos.shape))


@partial(jax.jit, static_argnums=(1, 3))
def move_goal_flip_walls(rng, params, state, n=1):
    if n == 0:
        return state

    def _mutate(carry, step):
        state = carry
        rng, mutation = step

        rng, arng, brng, crng, drng, erng = jax.random.split(rng, 6)

        is_flip_wall = jnp.equal(mutation, Mutations.FLIP_WALL.value)
        mutated_state = flip_wall(arng, state)
        next_state = jax.tree_map(lambda x, y: jax.lax.select(
            is_flip_wall, x, y), mutated_state, state)

        is_move_goal = jnp.equal(mutation, Mutations.MOVE_GOAL.value)
        mutated_state = move_goal(brng, state)
        next_state = jax.tree_map(lambda x, y: jax.lax.select(
            is_move_goal, x, y), mutated_state, next_state)

        is_move_pot = jnp.equal(mutation, Mutations.MOVE_POT.value)
        mutated_state = move_pot(crng, state)
        next_state = jax.tree_map(lambda x, y: jax.lax.select(
            is_move_pot, x, y), mutated_state, next_state)

        is_move_onion_pile = jnp.equal(
            mutation, Mutations.MOVE_ONION_PILE.value)
        mutated_state = move_onion_pile(drng, state)
        next_state = jax.tree_map(lambda x, y: jax.lax.select(
            is_move_onion_pile, x, y), mutated_state, next_state)

        is_move_plate_pile = jnp.equal(
            mutation, Mutations.MOVE_PLATE_PILE.value)
        mutated_state = move_bowl_pile(erng, state)
        next_state = jax.tree_map(lambda x, y: jax.lax.select(
            is_move_plate_pile, x, y), mutated_state, next_state)

        return next_state, None

    rng, nrng, *mrngs = jax.random.split(rng, n+2)
    mutations = jax.random.choice(nrng, np.arange(len(Mutations)), (n,))

    state, _ = jax.lax.scan(_mutate, state, (jnp.array(mrngs), mutations))

    onion_pos = jnp.zeros(state.wall_map.shape, dtype=jnp.uint8)
    plate_pos = jnp.zeros(state.wall_map.shape, dtype=jnp.uint8)
    dish_pos = jnp.zeros(state.wall_map.shape, dtype=jnp.uint8)

    pot_status = jnp.ones((state.wall_map.reshape(-1).shape), dtype=jnp.uint8) * 23

    next_maze_map = make_overcooked_map(
        state.wall_map,
        state.goal_pos,
        state.agent_pos,
        state.agent_dir_idx,
        state.bowl_pile_pos,
        state.onion_pile_pos,
        state.pot_pos,
        pot_status,
        onion_pos,
        plate_pos,
        dish_pos,
        pad_obs=True,
        num_agents=2,
        agent_view_size=5
    )

    return state.replace(maze_map=next_maze_map)


# Register the mutators
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register_mutator(env_id='Overcooked', mutator_id=None,
                 entry_point=module_path + ':move_goal_flip_walls')
