# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers
from domainbed.utils_fr import save_elapsed_time
from domainbed.scripts.generate_slurm_sweep import generate_slurm_sweep, generate_slurm_sweep_dalia

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['uv', 'run', '--no-sync', 'python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if v is None:
                continue

            # Boolean flags
            if isinstance(v, bool):
                if v:
                    command.append(f'--{k}')
                continue

            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)

            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps, checkpoint_freq, data_dir, task, holdout_fraction, uda_holdout_fraction, single_test_envs, hparams, dg_exps_dir, finetuned_exps_dir, selection_method_DA, save_emb_from_frozen, save_emb_from_finetuned): 
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if single_test_envs:
                    all_test_envs = [
                        [i] for i in range(datasets.num_environments(dataset))]
                else:
                    all_test_envs = all_test_env_combinations(
                        datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['uda_holdout_fraction'] = uda_holdout_fraction
                        train_args['dg_exps_dir'] = dg_exps_dir
                        train_args['finetuned_exps_dir'] = finetuned_exps_dir
                        train_args['selection_method_DA'] = selection_method_DA
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['checkpoint_freq'] = checkpoint_freq
                        train_args['trial_seed'] = trial_seed
                        train_args['save_emb_from_frozen'] = save_emb_from_frozen
                        train_args['save_emb_from_finetuned'] = save_emb_from_finetuned
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--single_test_envs', action='store_true')
    parser.add_argument('--save_emb_from_frozen', action='store_true')
    parser.add_argument('--save_emb_from_finetuned', action='store_true')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--dg_exps_dir', type=str, default=None)
    parser.add_argument('--finetuned_exps_dir', type=str, default=None)
    parser.add_argument('--selection_method_DA', type=str, default="IIDAccuracy")
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        checkpoint_freq=args.checkpoint_freq,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        uda_holdout_fraction=args.uda_holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams,
        dg_exps_dir = args.dg_exps_dir,
        finetuned_exps_dir = args.finetuned_exps_dir,
        selection_method_DA = args.selection_method_DA,
        save_emb_from_frozen = args.save_emb_from_frozen,
        save_emb_from_finetuned = args.save_emb_from_finetuned
    )

    # Add selection method in paths
    if "adaptation" in args.task or args.save_emb_from_finetuned:
        args.output_dir = os.path.join(args.output_dir, args.selection_method_DA)

    # create job list  
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    active_jobs = [
    job for job in jobs
    if job.state in [Job.NOT_LAUNCHED, Job.INCOMPLETE]
    ]

    to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
    # Job.delete(to_delete)
    
    

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    # generate array slurm launcher try
    # generate_slurm_sweep(args, active_jobs, sweep_output_dir=args.output_dir, job_name="Test")
    generate_slurm_sweep_dalia(args, active_jobs, sweep_output_dir=args.output_dir, job_name="Test")
    print(f"Total jobs: {len(jobs)}")
    print(f"Jobs à lancer: {len(active_jobs)}")
    exit(0)

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
        generate_slurm_sweep(args, to_delete, sweep_output_dir=args.output_dir, job_name="Test", role="incomplete", device="h100", account="ldz@h100")
