"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re
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
from minimax.util.rl import AgentPopHeterogenous
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
    '--pckl_path',
    type=str,
    default=None,
    help='Path to population json file.')
parser.add_argument(
    '--meta_path',
    type=str,
    default=None,
    help='Path to population json file.')
parser.add_argument(
    '--log_dir',
    type=str,
    default='~/logs/minimax',
    help='Log directory containing experiment dirs.')
parser.add_argument(
    '--other_pckl_path',
    type=str,
    default=None,
    help='Path to population json file.')
parser.add_argument(
    '--other_meta_path',
    type=str,
    default=None,
    help='Path to population json file.')
parser.add_argument(
    '--env_names',
    type=str,
    help='csv of evaluation environments.')
parser.add_argument(
    '--n_episodes',
    type=int,
    default=1,
    help='Number of evaluation episodes.')
parser.add_argument(
    '--agent_idxs',
    type=str,
    default='*',
    help="Indices of agents to evaluate. '*' indicates all.")
parser.add_argument(
    '--render_mode',
    type=str,
    nargs='?',
    const=True,
    default=None,
    help='Visualize episodes.')
parser.add_argument(
    '--results_path',
    type=str,
    default='results/',
    help='Results dir.')
parser.add_argument(
    '--results_fname',
    type=str,
    default=None,
    help='Results filename (without .csv).')


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

    # log_dir_path = os.path.expandvars(os.path.expanduser(args.log_dir))

    all_eval_stats = defaultdict(list)
    # xpid_dir_path = os.path.join(log_dir_path, xpid)
    checkpoint_path = args.pckl_path
    meta_path = args.meta_path

    other_agent_checkpoint_path = args.other_pckl_path
    other_agent_meta_path = args.other_meta_path

    # Load checkpoint info
    if not os.path.exists(meta_path):
        print(f'Configuration at {meta_path} does not exist. Skipping...')

    if not os.path.exists(other_agent_meta_path):
        raise ValueError(f"Did not find: {other_agent_meta_path}")

    if not os.path.exists(checkpoint_path):
        print(
            f'Checkpoint path {checkpoint_path} does not exist. Skipping...')

    if not os.path.exists(other_agent_checkpoint_path):
        raise ValueError(f"Did not find: {other_agent_checkpoint_path}")

    xp_args = load_config(meta_path)

    xp_other_args = load_config(other_agent_meta_path)

    agent_idxs = args.agent_idxs
    if agent_idxs == '*':
        agent_idxs = np.arange(xp_args.train_runner_args.n_students)
    else:
        agent_idxs = \
            np.array([int(x) for x in agent_idxs.split(',')])
        assert np.max(agent_idxs) <= xp_args.train_runner_args.n_students, \
            'Agent index is out of bounds.'

    runner_state_0 = load_pkl_object(checkpoint_path)
    if "params" in runner_state_0[1].keys():
        params_0 = runner_state_0[1]['params']
    elif "actor_params" in runner_state_0[1].keys():
        params_0 = runner_state_0[1]['actor_params']
    else:
        raise ValueError("No params found in checkpoint.")

    params_0 = jax.tree_util.tree_map(
        lambda x: jnp.take(x, indices=agent_idxs, axis=0),
        params_0
    )

    xp_args_other = load_config(other_agent_meta_path)

    runner_state_1 = load_pkl_object(other_agent_checkpoint_path)
    if "params" in runner_state_1[1].keys():
        params_1 = runner_state_1[1]['params']
    elif "actor_params" in runner_state_1[1].keys():
        params_1 = runner_state_1[1]['actor_params']
    else:
        raise ValueError("No params found in checkpoint.")

    params_1 = jax.tree_util.tree_map(
        lambda x: jnp.take(x, indices=agent_idxs, axis=0),
        params_1
    )

    for i in range(2):
        # Swap params and runner states
        # Bit finicky be careful here
        if i == 1:
            params_0, params_1 = params_1, params_0
            xp_args, xp_other_args = xp_other_args, xp_args
            runner_state_0, runner_state_1 = runner_state_1, runner_state_0

        with jax.disable_jit(args.render_mode is not None):
            student_model = models.make(
                env_name=xp_args.env_name,
                model_name=xp_args.student_model_name,
                **xp_args.student_model_args
            )

            population_model = models.make(
                env_name=xp_args.env_name,
                model_name=xp_other_args.student_model_name,
                **xp_other_args.student_model_args
            )

            # We force EvalRunner to select all params, since we've already
            # extracted the relevant agent indices.
            if "Overcooked" in args.env_names:
                from minimax.runners_ma import EvalRunnerHeterogenous

                pop = AgentPopHeterogenous(
                    agent_0=agents.MAPPOAgent(
                        actor=student_model, critic=None),
                    agent_1=agents.MAPPOAgent(
                        actor=population_model, critic=None),
                    n_agents=len(agent_idxs)
                )
            else:
                raise ValueError("Unknown environment.")

            runner = EvalRunnerHeterogenous(
                pop=pop,
                env_names=args.env_names,
                env_kwargs=xp_args.eval_env_args,
                n_episodes=args.n_episodes,
                render_mode=args.render_mode,
                agent_idxs='*'
            )

            rng = jax.random.PRNGKey(args.seed)
            _eval_stats = runner.run(rng, params_0, params_1)

            eval_stats = {}
            for k, v in _eval_stats.items():
                prefix_match = re.match(r'^eval/(a[0-9]+):.*', k)
                if prefix_match is not None:
                    prefix = prefix_match.groups()[0]
                    _idx = int(prefix.lstrip('a').rstrip(':'))
                    idx = agent_idxs[_idx]
                    new_prefix = f'a{idx}'
                    new_k = k.replace(prefix, new_prefix)
                    eval_stats[new_k] = v
                else:
                    eval_stats[k] = v

            for k, v in eval_stats.items():
                all_eval_stats[k].append(float(v))

    aggregate_eval_stats = {}
    for k, v in all_eval_stats.items():
        max = np.max(all_eval_stats[k])
        mean = np.mean(all_eval_stats[k])
        if len(all_eval_stats[k]) > 1:
            sem = spstats.sem(all_eval_stats[k])
        else:
            sem = 0.0
        aggregate_eval_stats[k] = f'{mean: 0.4}+/-{sem: 0.4} (max: {max: 0.4})'

        _min = np.min(all_eval_stats[k])
        aggregate_eval_stats[f'min:{k}'] = f'{_min: 0.4}'

    logger = HumanOutputFormat(sys.stdout)
    logger.writekvs(aggregate_eval_stats)

    if args.results_fname is not None:
        if args.results_fname.strip('"') == '*':
            results_fname = args.xpid_prefix or args.xpid
        else:
            results_fname = args.results_fname

        df = pd.DataFrame.from_dict(all_eval_stats)
        results_path = args.results_path
        if not os.path.isabs(results_path):
            results_path = os.path.join(
                os.path.abspath(__file__), results_path)
        results_path = os.path.join(results_path, f'{results_fname}.csv')
        df.to_csv(results_path, index=False)
        print(f'Saved results to {results_path}')
