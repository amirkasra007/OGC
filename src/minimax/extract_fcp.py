"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import glob
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

parser.add_argument(
    '--trained_seed',
    type=int,
    default=None,
    help='Seed that the model was trained with')

if __name__ == '__main__':
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

        sub_checkpoint_paths = glob.glob(f"{checkpoint_path[:-4]}*.pkl")
        sub_checkpoint_paths = sorted(list(sub_checkpoint_paths))

        map_name_path = {}
        map_name_params = {}
        for sub_checkpoint_path in sub_checkpoint_paths:
            desc = sub_checkpoint_path[len(checkpoint_path[:-4])+1:-4]
            if desc == '':
                desc = 'final'
            runner_state = load_pkl_object(sub_checkpoint_path)
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
            map_name_path[desc] = sub_checkpoint_path
            map_name_params[desc] = params

        map_name_eval_stas = {}
        for desc, params in map_name_params.items():
            with jax.disable_jit(args.render_mode is not None):
                student_model = models.make(
                    env_name=xp_args.env_name,
                    model_name=xp_args.student_model_name,
                    **xp_args.student_model_args
                )

                # We force EvalRunner to select all params, since we've already
                # extracted the relevant agent indices.
                if "Overcooked" in args.env_names:
                    from minimax.runners_ma import EvalRunner

                    pop = AgentPop(
                        agent=agents.MAPPOAgent(
                            actor=student_model, critic=None),
                        n_agents=len(agent_idxs)
                    )
                elif "Maze" in args.env_names:
                    from minimax.runners import EvalRunner

                    pop = AgentPop(
                        agent=agents.PPOAgent(model=student_model),
                        n_agents=len(agent_idxs)
                    )
                else:
                    raise ValueError("Unknown environment.")

                runner = EvalRunner(
                    pop=pop,
                    env_names=args.env_names,
                    env_kwargs=xp_args.eval_env_args,
                    n_episodes=args.n_episodes,
                    render_mode=args.render_mode,
                    agent_idxs='*'
                )

                rng = jax.random.PRNGKey(args.seed)
                _eval_stats = runner.run(rng, params)

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

                pbar.update(1)

                assert len(
                    runner.ext_env_names) == 1, "Only one at a time to avoid confusion!"

                map_name_eval_stas[desc] = eval_stats[
                    f"eval/a0:test_return:{runner.ext_env_names[0]}"]

        best = max(map_name_eval_stas.items(), key=lambda x: x[1])
        mid = min(map_name_eval_stas.items(),
                  key=lambda x: abs(x[1]-int(best[1])/2))
        low = min(map_name_eval_stas.items(),
                  key=lambda x: abs(x[1]-int(best[1])/5))

        print("\n\n------------------------------------\n\n")
        print("Best: ", best)
        print("Mid: ", mid)
        print("Low: ", low)

        high_id = best[0]
        mid_id = mid[0]
        low_id = low[0]

        # Take paths from ids and cp files to /populations/fcp/seed/
        high_path = map_name_path[high_id]
        mid_path = map_name_path[mid_id]
        low_path = map_name_path[low_id]

        print("High path: ", high_path)
        print("Mid path: ", mid_path)
        print("Low path: ", low_path)

        # find seed string with SEED_*_ in xpid
        seed = args.trained_seed

        target_dir = f"{os.getcwd()}/populations/fcp/{args.env_names}/{seed}/"

        print("Target dir: ", target_dir)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        os.system(f"cp {high_path} {target_dir}high.pkl")
        os.system(f"cp {mid_path} {target_dir}mid.pkl")
        os.system(f"cp {low_path} {target_dir}low.pkl")
        # also copy meta
        os.system(f"cp {meta_path} {target_dir}meta.json")

        # make a txt file there and copy the xpid
        with open(f"{target_dir}xpid.txt", "w") as f:
            f.write(xpid)

    pbar.close()
