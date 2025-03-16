"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import re
import fnmatch
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as spstats
import jax
import jax.numpy as jnp
from tqdm import tqdm

from minimax.util.parsnip import Parsnip
from minimax.util.checkpoint import load_pkl_object, load_config
from minimax.util.loggers import HumanOutputFormat
from minimax.util.rl import AgentPop
import minimax.models as models
import minimax.agents as agents


parser = Parsnip()

# ==== Define top-level arguments
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    help='Random seed.')
parser.add_argument(
    '--log_dir',
    type=str,
    default='~/logs/minimax',
    help='Log directory containing experiment dirs.')
parser.add_argument(
    '--xpid',
    type=str,
    default='latest',
    help='Experiment ID dir name for model.')
parser.add_argument(
    '--xpid_prefix',
    type=str,
    default=None,
    help='Experiment ID dir name for model.')
parser.add_argument(
    '--checkpoint_name',
    type=str,
    default='checkpoint',
    help='Name of checkpoint .pkl.')
parser.add_argument(
    '--agent_idxs',
    type=str,
    default='*',
    help="Indices of agents to evaluate. '*' indicates all.")


if __name__ == '__main__':
    """
    Usage: 
            python -m eval \
            --xpid= \
            --env_names="Maze-SixteenRooms" \
            --n_episodes=100 \
            --agent_idxs=0
    """
    args = parser.parse_args()

    log_dir_path = os.path.expandvars(os.path.expanduser(args.log_dir))

    xpids = []
    if args.xpid_prefix is not None:
        # Get all matching xpid directories
        all_xpids = fnmatch.filter(os.listdir(
            log_dir_path), f"{args.xpid_prefix}*")
        filter_re = re.compile('.*_[0-9]*$')
        xpids = [x for x in all_xpids if filter_re.match(x)]
    else:
        xpids = [args.xpid]

    pbar = tqdm(total=len(xpids))

    all_eval_stats = defaultdict(list)
    for xpid in xpids:
        xpid_dir_path = os.path.join(log_dir_path, xpid)
        checkpoint_path = os.path.join(
            xpid_dir_path, f'{args.checkpoint_name}.pkl')
        meta_path = os.path.join(xpid_dir_path, f'meta.json')

        # Load checkpoint info
        if not os.path.exists(meta_path):
            print(f'Configuration at {meta_path} does not exist. Skipping...')
            continue

        if not os.path.exists(checkpoint_path):
            print(
                f'Checkpoint path {checkpoint_path} does not exist. Skipping...')
            continue

        xp_args = load_config(meta_path)

        agent_idxs = args.agent_idxs
        if agent_idxs == '*':
            agent_idxs = np.arange(xp_args.train_runner_args.n_students)
        else:
            agent_idxs = \
                np.array([int(x) for x in agent_idxs.split(',')])
            assert np.max(agent_idxs) <= xp_args.train_runner_args.n_students, \
                'Agent index is out of bounds.'

        runner_state = load_pkl_object(checkpoint_path)
        if "params" in runner_state[1].keys():
            params = runner_state[1]['params']
        elif "actor_params" in runner_state[1].keys():
            params = runner_state[1]['actor_params']
        else:
            raise ValueError("No params found in checkpoint.")

        params = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices=agent_idxs, axis=0),
            params
        )

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Model has {param_count} parameters.")
