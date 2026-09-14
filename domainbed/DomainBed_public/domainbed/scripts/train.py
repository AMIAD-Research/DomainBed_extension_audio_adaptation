# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed.lib import adaptation
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.utils_fr import save_accuracy_per_class, statistics_splits, extract_embeddings, compute_jacobian_norm
# from domainbed.scripts_audio.inference_only import noise_inference
from domainbed import model_selection
import copy

if __name__ == "__main__":
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "supervised_domain_adaptation", "unsupervised_domain_adaptation", "mono_source"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0],
        help='chose the index of the dataset not used during training') 
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--dg_exps_dir', type=str, default=None, help="directoryf from which we load optimal hparams from hparams grid search in DG")
    parser.add_argument('--selection_method_DA', type=str, default="IIDAccuracy")
    parser.add_argument('--save_emb_from_frozen', action='store_true')
    parser.add_argument('--save_emb_from_finetuned', action='store_true')
    parser.add_argument('--finetuned_exps_dir', type=str, default=None, help="directory from which we load selected checkpoints")

    args = parser.parse_args()

    # Mono-source exps
    real_test_env = args.test_envs
    all_envs = [0,1,2,3]
    print(f"Real test env {real_test_env}")

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    writer_board = SummaryWriter(log_dir=args.output_dir)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # tr_dataset = args.uda_holdout_fraction # Choose size of global dataset 
    # train only on clean
    # args.test_envs = [1,2] # Choose on which domain we train 0:clean, 1:wham, 2:wind 3:saturation
    # tr_env = [0]
    # Modifier le trainloader lorsqu'on supprimera cela !! 

    if "domain_adaptation" in args.task: 
        # chose hparams from Domain Generalization hparams search. 
        # Import for comparing results between DG and DA
        hparams = adaptation.load_hparams_from_dg(args, hparams)  
    # if "mono_source" in args.task: 
    #     # chose hparams from Domain Generalization hparams search. 
    #     hparams = adaptation.load_hparams_from_dg(args, hparams)  
    elif "linear_probing" in args.task: 
        hparams = adaptation.load_hparams_from_dg(args, hparams)   
        hparams['freeze_all'] = True 
        args.task = "domain_generalization"

    if args.save_emb_from_frozen: 
        hparams = adaptation.load_hparams_from_dg(args, hparams)   
        hparams['freeze_all'] = True
        train_envs = [x for x in all_envs if x not in args.test_envs]
    
    elif args.save_emb_from_finetuned:
        hparams = adaptation.load_hparams_from_dg(args, hparams) 
        checkpoint_dir = adaptation.load_finetuned_checkpoint(args, hparams)
        print(f"Loading checkpoint from {checkpoint_dir}")
        train_envs = [x for x in all_envs if x not in args.test_envs]  

    ## Mono source experiments
    # args.dg_exps_dir = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/DG/erm"
    # args.selection_method_DA = "IIDAccuracy"
    # print(f"test env loading hparams {args.test_envs}")
    # hparams = adaptation.load_hparams_from_dg(args, hparams)   
    
    # args.test_envs = [x for x in all_envs if x not in args.test_envs]
    # print(f"test env loading data {args.test_envs} (should change to all except real test env)") 
    
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        if (args.dataset == "noisy_CWWS_train_small") or (args.dataset == "noisy_CWWS_train_mono"):
            dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)

            dataset_validation = vars(datasets)["noisy_CWWS_validation"](args.data_dir,
            args.test_envs, hparams)

            dataset_test = vars(datasets)["noisy_CWWS_test"](args.data_dir,
            args.test_envs, hparams)
        else:
            dataset = vars(datasets)[args.dataset](args.data_dir,
                args.test_envs, hparams)
    else:
        raise NotImplementedError
    
    if args.task == "mono_source":
        args.test_envs = [x for x in all_envs if x not in args.test_envs]
        print(f"test env loading data {args.test_envs} (should change to all except real test env)") 

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_all_splits = [] # To be removed
    
    if "domain_adaptation" in args.task:
        uda_splits = []
    
    for env_i, env in enumerate(dataset):
        # uda = []
        # uda_all = []

        # create the out_split from all env data
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        # Utiliser stratified_split_dataset
        
        ##### Study size global dataset - to be removed
        # in_, _ = misc.split_dataset(env,
        #     int(len(env)*tr_dataset),
        #     misc.seed_hash(args.trial_seed, env_i))


        # Split uda and in_split in two equal parts. Only uda from test_envs will be loaded in uda_loader (cf. uda_loaders implementation with "if" condition). The goal is to have the same proportion of in_split data for all DA and DG experiments. Then a fraction of uda_all is selected for training in test_envs.
        # UDA_ALL AND IN_ have the same number of data ! 
        uda_all, in_ = misc.split_dataset(in_,
                int(len(in_)*0.5),
                misc.seed_hash(args.trial_seed, env_i))
        
        # print(f"{len(in_)} samples in the training set")
        ### Study size global dataset - to be removed
        # uda_all, _ = misc.split_dataset(in_,
        #         int(len(in_)*0.5),
        #         misc.seed_hash(args.trial_seed, env_i))             
        
        # print(f"The train split of each domain contains {int(len(env)*tr_dataset)} samples")

        # Create uda splits for all environments. The selection of test_envs is done in the loader. 
        if "domain_adaptation" in args.task:
            # uda, _ = misc.split_dataset(uda_all,
            #     int(len(uda_all)*args.uda_holdout_fraction),
            #     misc.seed_hash(args.trial_seed, env_i)) 
            if args.uda_holdout_fraction > 1.1: 
                uda, _ = misc.stratified_split_dataset(uda_all,
                    int(len(uda_all)*1),
                    misc.seed_hash(args.trial_seed, env_i)) 
            else: 
                uda, _ = misc.stratified_split_dataset(uda_all,
                    int(len(uda_all)*args.uda_holdout_fraction),
                    misc.seed_hash(args.trial_seed, env_i)) 

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        
        uda_all_splits.append((uda_all, uda_weights)) 
    
        if "domain_adaptation" in args.task:
            uda_splits.append((uda, uda_weights))

        # if len(uda):
        #     uda_splits.append((uda, uda_weights))
        # if len(uda_all): # To be removed
            # uda_all_splits.append((uda_all, uda_weights))
    
    if args.uda_holdout_fraction > 1.1 and args.dataset == "noisy_CWWS_train_small":
            uda_big_splits = []
            uda_dataset = vars(datasets)["noisy_CWWS_train_mono"](args.data_dir,
            args.test_envs, hparams)

            for env_i, env in enumerate(uda_dataset):
                big_uda_all,_ = misc.split_dataset(env,
                int(len(env)),
                misc.seed_hash(args.trial_seed, env_i))

                if hparams['class_balanced']:
                    in_weights = misc.make_weights_for_balanced_classes(in_)
                    out_weights = misc.make_weights_for_balanced_classes(out)
                    if uda is not None:
                        uda_weights = misc.make_weights_for_balanced_classes(uda)
                else:
                    in_weights, out_weights, uda_weights = None, None, None

                uda_big_splits.append((big_uda_all, in_weights))

    if (args.dataset == "noisy_CWWS_train_small") or (args.dataset == "noisy_CWWS_train_mono"):
        in_val_splits = []
        out_val_splits = []
        for env_i, env in enumerate(dataset_validation):
            out_val, in_val = misc.split_dataset(env,
            int(len(env)),
            misc.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None

            in_val_splits.append((in_val, in_weights))
            out_val_splits.append((out_val, out_weights))

        in_test_splits = []
        out_test_splits = []
        for env_i, env in enumerate(dataset_test):
            out_test, in_test = misc.split_dataset(env,
            int(len(env)*0),
            misc.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
                if uda is not None:
                    uda_weights = misc.make_weights_for_balanced_classes(uda)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            
            in_test_splits.append((in_test, in_weights))
            out_test_splits.append((out_test, out_weights))

    # if "domain_adaptation" in args.task:
    #     statistics_splits(args.output_dir, in_splits, out_splits, uda_all_splits, uda_splits)
    #     if (args.dataset == "noisy_CWWS_train_small") or (args.dataset == "noisy_CWWS_train_mono"):
    #         print(f"\n=====  Test and validation sets for {args.dataset} dataset =====\n")
    #         if args.uda_holdout_fraction > 1.1: 
    #             statistics_splits(args.output_dir, in_test_splits, out_val_splits, uda_big_splits, uda_big_splits)
    #         else: 
    #             statistics_splits(args.output_dir, in_test_splits, out_val_splits, uda_all_splits, uda_splits)

    # else:
    #     statistics_splits(args.output_dir, in_splits, out_splits, uda_all_splits)
    #     if (args.dataset == "noisy_CWWS_train_small") or (args.dataset == "noisy_CWWS_train_mono"):
    #         print(f"\n=====  Test and validation sets for {args.dataset} dataset =====\n\n")
    #         statistics_splits(args.output_dir, in_test_splits, out_val_splits, uda_all_splits)

    if "domain_adaptation" in args.task and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # print("before loaders")
    print(in_splits)
    print(args.test_envs)
    # breakpoint()
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    finite_train_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
        # if i in tr_env]
    print("done with train loader")

    print(train_loaders)
    print(args.test_envs)
    train_total_samples = sum(len(env) 
                    for i, (env, _) in enumerate (in_splits)
                    if i not in args.test_envs)
    print(train_total_samples)
    print("done with train")

    # To be removed 
    # uda_all_loaders = [InfiniteDataLoader( 
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_all_splits)
    #     if i in args.test_envs]

    # print(uda_all_loaders)
    # uda_all_total_samples = sum(len(env) 
    #                 for i, (env, env_weights) in enumerate(uda_all_splits)
    #                 if i in args.test_envs)
    # print(uda_all_total_samples)
    # print("done with uda all")
    
    if "domain_generalization" in args.task or args.uda_holdout_fraction == 0:
        uda_splits = []
    
    if args.uda_holdout_fraction >= 1.1 and args.dataset == "noisy_CWWS_train_small":
        uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=min(hparams['batch_size'], len(env)),
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_big_splits)
            if i in args.test_envs]
        
        finite_uda_loaders = [FastDataLoader(
            dataset=env,
            batch_size=min(hparams['batch_size'], len(env)),
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_big_splits)
            if i in args.test_envs]
    else: 
        uda_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=min(hparams['batch_size'], len(env)),
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)
            if i in args.test_envs]
        
        finite_uda_loaders = [FastDataLoader(
            dataset=env,
            batch_size=min(hparams['batch_size'], len(env)),
            num_workers=dataset.N_WORKERS)
            for i, (env, env_weights) in enumerate(uda_splits)
            if i in args.test_envs]
        
    print("done with uda loader")
    # uda_total_samples = sum(len(env) 
    #                 for i, (env, env_weights) in enumerate(uda_splits)
    #                 if i in args.test_envs)
    # print(uda_total_samples)
    # print("done with uda")

    
    if (args.dataset == "noisy_CWWS_train_small") or (args.dataset == "noisy_CWWS_train_mono"):
        print(f'Eval on external test dataset and validate on external validation dataset')
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS)
            for env, _ in (in_test_splits + out_val_splits + uda_splits)]

    else:
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits + uda_splits)]
    
    print("done with eval loader")
    
    # eval_total_samples = sum(len(env) 
    #                 for env, _ in (in_splits + out_splits + uda_splits))
    # print(eval_total_samples)
    # print("done with eval")

    # Mono-source exps
    args.test_envs = real_test_env
    print(f"Back to real test env {args.test_envs}")

    # python -m domainbed.scripts.train --data_dir=$SCRATCH/domainbed/data/ --algorithm ERM --dataset AUDIO_DA --test_env 1 --output_dir $SCRATCH/domainbed/results/debug_UDA --uda_holdout_fraction 0.01
    # Après ces vérifications, lancer les premiers sweep pour vérifier que tout se passe correctement PUIS Lancement à grande échelle des expériences DG !! 
    # Lors de ces entrainments, effectuer l'implémentation de DA et UDA. ATTENTION: lorsque les sweep tournent, la modification du code se transmet aux train qui vont se lancer dans el sweep !! Trouver une méthode pour décorreler les codes entre DG qui tourne et celui de travail. 

    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)

    ## To be removed: use it only at inference
    # ev = (eval_loader_names, eval_loaders, eval_weights)
    # if "wind" in args.output_dir:
    #     noise_inference(algorithm=algorithm, evaluate=ev, device=device, noise_type="wind")
    # elif "wham" in args.output_dir: 
    #     noise_inference(algorithm=algorithm, evaluate=ev, device=device, noise_type="wham")
    # elif "saturation" in args.output_dir: 
    #     noise_inference(algorithm=algorithm, evaluate=ev, device=device, noise_type="saturation")
    # else: 
    #     raise ValueError(f"Unkown noise type")
    #

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
        
    if args.save_emb_from_frozen:

        frozen_featurizer = copy.deepcopy(algorithm.featurizer)
        frozen_classifier = copy.deepcopy(algorithm.classifier)

        frozen_featurizer.eval()
        frozen_classifier.eval()

        for p in frozen_featurizer.parameters():
            p.requires_grad = False

        for p in frozen_classifier.parameters():
            p.requires_grad = False

    elif args.save_emb_from_finetuned:

        model = copy.deepcopy(algorithm)


        model_path = os.path.join(checkpoint_dir, "model.pkl")



        model_path = Path(model_path)

        if not model_path.exists():
            parts = model_path.parts

            try:
                results_idx = parts.index("results")
            except ValueError:
                raise ValueError(f"'results' not found in model path: {model_path}")

            new_base_path = Path(os.environ["SCRATCH"]) / "domainbed"
            model_path = new_base_path.joinpath(*parts[results_idx:])

            # If the reconstructed path does not exist, try replacing CWWS_dal by CWWS
            if not model_path.exists():
                model_path = Path(str(model_path).replace("CWWS_dal", "CWWS"))

        print(model_path)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False,)


        model.load_state_dict(checkpoint["model_dict"])
        model.to(device)
        model.eval()

        finetuned_featurizer = copy.deepcopy(model.featurizer)
        finetuned_classifier = copy.deepcopy(model.classifier)
        finetuned_model = copy.deepcopy(model)

        finetuned_featurizer.eval()
        finetuned_classifier.eval()

        for p in finetuned_featurizer.parameters():
            p.requires_grad = False

        for p in finetuned_classifier.parameters():
            p.requires_grad = False


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]

        if "domain_adaptation" in args.task:
            if args.uda_holdout_fraction == 0: 
                da_device = None
            elif args.task == "unsupervised_domain_adaptation":
                da_device = [x.to(device)
                    for x,_ in next(uda_minibatches_iterator)]
            elif args.task == "supervised_domain_adaptation":
                da_device = [(x.to(device), y.to(device))
                    for x,y in next(uda_minibatches_iterator)]
        else:
            da_device = None

        step_vals = algorithm.update(minibatches_device, da_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val) 

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):

            # log feature space for analysis
            features = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            results_vs_random = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            acc_random = 0.335

            results_per_class = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            results_std = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            
            
            # Tensorboard 
            for key, value in results.items():
                if "loss" in key.lower():
                    writer_board.add_scalar(
                        tag=f"loss/{key}_vs_step",
                        scalar_value=value,
                        global_step=results["step"]
                    )
                    writer_board.add_scalar(
                        tag=f"loss/{key}_vs_epoch",
                        scalar_value=value,
                        global_step=results["epoch"]
                    )
            eval_env_indices = (
                    list(range(len(in_splits)))
                    + list(range(len(out_splits)))
                    + list(range(len(uda_splits)))
                )
            
            evals = zip(
                eval_loader_names,
                eval_loaders,
                eval_weights,
                eval_env_indices
            )
            for name, loader, weights, env_idx in evals:
                acc, var = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                results_vs_random[name+'acc_vs_rd'] = acc - acc_random
                results_std[name+'_var'] = var
                acc_per_class = misc.accuracy_per_class(algorithm, loader, weights, device)
                for cls, v in acc_per_class.items():
                    results_per_class[f"{name}/acc_class_{cls}"] = v
                
                if (args.save_emb_from_frozen or args.save_emb_from_finetuned) and step == 0 and name.endswith("_in"):

                    if args.save_emb_from_frozen:
                        featurizer = frozen_featurizer.eval()
                        classifier = frozen_classifier.eval()
                    elif args.save_emb_from_finetuned: 
                        featurizer = finetuned_featurizer.eval()
                        classifier = finetuned_classifier.eval()
                        model = finetuned_model

                    jacobian_values = None

                    if args.save_emb_from_finetuned:

                        jacobian_values = []

                        for x, _ in loader:

                            x = x.to(device)

                            jac = compute_jacobian_norm(
                                model,
                                x
                            )

                            jacobian_values.append(
                                jac.detach().cpu()
                            )

                        jacobian_values = torch.cat(
                            jacobian_values
                        )

                    embeddings = []
                    logits_list = []
                    probabilities = []
                    labels = []

                    with torch.no_grad():
                        for x, y in loader:

                            x = x.to(device)

                            z = featurizer(x)

                            logits = classifier(z)
                            probs = torch.softmax(logits, dim=1)

                            embeddings.append(z.cpu())
                            logits_list.append(logits.cpu())
                            probabilities.append(probs.cpu())
                            labels.append(y.cpu())

                    embeddings = torch.cat(embeddings, dim=0)
                    logits_list = torch.cat(logits_list, dim=0)
                    probabilities = torch.cat(probabilities, dim=0)
                    labels = torch.cat(labels, dim=0)

                    is_target = env_idx in args.test_envs

                    if is_target:
                        filename = f"target_{env_idx}.pt"
                    else:
                        filename = f"source_{env_idx}.pt"

                    torch.save(
                        {
                            "embeddings": embeddings,
                            "logits": logits_list,
                            "probabilities": probabilities,
                            "labels": labels,
                            "env": env_idx,
                            "is_target": is_target,
                            "jacobian_norm": jacobian_values,
                            "jacobian_norm_mean": jacobian_values.mean().item(),
                            "jacobian_norm_std": jacobian_values.std().item(),
                        },
                        os.path.join(args.output_dir, filename)
                    )

            if (args.save_emb_from_frozen or args.save_emb_from_finetuned) and step == 0:

                if args.save_emb_from_frozen:
                    featurizer = frozen_featurizer.eval()
                    classifier = frozen_classifier.eval()
                elif args.save_emb_from_finetuned: 
                    featurizer = finetuned_featurizer.eval()
                    classifier = finetuned_classifier.eval()
                    model = finetuned_model

                used_train_embeddings = []
                used_train_logits = []
                used_train_probabilities = []
                used_train_labels = []
                used_train_envs = []
                used_train_is_target = []
                used_train_jacobian_norm = []
                

                for env_idx, loader in zip(train_envs, finite_train_loaders):

                    if args.save_emb_from_frozen:
                        print(
                            f"Extracting frozen embeddings from source env {env_idx}..."
                        )
                    elif args.save_emb_from_finetuned:
                        print(
                            f"Extracting finetuned embeddings from source env {env_idx}..."
                        )

                    embeddings = []
                    logits_list = []
                    probabilities = []
                    labels = []
                    jacobian_norms = []

                    for x, _ in loader:

                        x = x.to(device)

                        jac = compute_jacobian_norm(
                            model,
                            x
                        )

                        jacobian_norms.append(
                            jac.detach().cpu()
                        )


                    jacobian_norms = torch.cat(
                        jacobian_norms,
                        dim=0
                    )

                    with torch.no_grad():
                        for x, y in loader:

                            x = x.to(device)
                            z = featurizer(x)

                            logits = classifier(z)
                            probs = torch.softmax(logits, dim=1)

                            logits_list.append(logits.cpu())
                            probabilities.append(probs.cpu())
                            embeddings.append(z.cpu())
                            labels.append(y.cpu())

                    embeddings = torch.cat(embeddings, dim=0)
                    logits_list = torch.cat(logits_list, dim=0)
                    probabilities = torch.cat(probabilities, dim=0)
                    labels = torch.cat(labels, dim=0)

                    # Embeddings
                    used_train_embeddings.append(embeddings)
                    # Logits
                    used_train_logits.append(logits_list)
                    used_train_probabilities.append(probabilities)

                    # Labels
                    used_train_labels.append(labels)

                    used_train_envs.append(
                        torch.full(
                            (len(labels),),
                            env_idx,
                            dtype=torch.long
                        )
                    )

                    used_train_is_target.append(
                        torch.zeros(
                            len(labels),
                            dtype=torch.bool
                        )
                    )

                    used_train_jacobian_norm.append(
                        jacobian_norms
                    )

                    print(
                        f"  env {env_idx}: {len(labels)} samples"
                    )

                if "domain_adaptation" in args.task:
                    for env_idx, loader in zip(args.test_envs, finite_uda_loaders):

                        if args.save_emb_from_frozen:
                            print(
                                f"Extracting frozen embeddings from target/UDA env {env_idx}..."
                            )
                        elif args.save_emb_from_finetuned:
                            print(
                                f"Extracting finetuned embeddings from target/UDA env {env_idx}..."
                            )

                        embeddings = []
                        logits_list = []
                        probabilities = []
                        labels = []
                        jacobian_norms = []

                        for x, _ in loader:

                            x = x.to(device)

                            jac = compute_jacobian_norm(
                                model,
                                x
                            )

                            jacobian_norms.append(
                                jac.detach().cpu()
                            )


                        jacobian_norms = torch.cat(
                            jacobian_norms,
                            dim=0
                        )

                        with torch.no_grad():
                            for x, y in loader:
                                x = x.to(device)

                                z = featurizer(x)
                                logits = classifier(z)
                                probs = torch.softmax(logits, dim=1)

                                embeddings.append(z.cpu())
                                logits_list.append(logits.cpu())
                                probabilities.append(probs.cpu())
                                labels.append(y.cpu())

                        embeddings = torch.cat(embeddings, dim=0)
                        logits_list = torch.cat(logits_list, dim=0)
                        probabilities = torch.cat(probabilities, dim=0)
                        labels = torch.cat(labels, dim=0)

                        used_train_embeddings.append(embeddings)
                        used_train_logits.append(logits_list)
                        used_train_probabilities.append(probabilities)
                        used_train_labels.append(labels)
                        used_train_jacobian_norm.append(
                            jacobian_norms
                        )

                        used_train_envs.append(
                            torch.full(
                                (len(labels),),
                                env_idx,
                                dtype=torch.long
                            )
                        )

                        used_train_is_target.append(
                            torch.ones(
                                len(labels),
                                dtype=torch.bool
                            )
                        )

                        print(
                            f"  UDA env {env_idx}: {len(labels)} samples"
                        )

                used_train_embeddings = torch.cat(
                    used_train_embeddings,
                    dim=0
                )

                used_train_logits = torch.cat(
                    used_train_logits,
                    dim=0
                )

                used_train_probabilities = torch.cat(
                    used_train_probabilities,
                    dim=0
                )

                used_train_labels = torch.cat(
                    used_train_labels,
                    dim=0
                )

                used_train_envs = torch.cat(
                    used_train_envs,
                    dim=0
                )

                used_train_is_target = torch.cat(
                    used_train_is_target,
                    dim=0
                )

                used_train_jacobian_norm = torch.cat(
                    used_train_jacobian_norm,
                    dim=0
                )

                used_train_path = os.path.join(
                    args.output_dir,
                    "used_train.pt"
                )

                torch.save(
                    {
                        "embeddings": used_train_embeddings,
                        "logits": used_train_logits,
                        "probabilities": used_train_probabilities,
                        "labels": used_train_labels,
                        "env": used_train_envs,
                        "is_target": used_train_is_target,
                        # Jacobian norm per sample
                        "jacobian_norm": used_train_jacobian_norm,

                        # Optional global summary
                        "jacobian_norm_mean": used_train_jacobian_norm.mean().item(),
                        "jacobian_norm_std": used_train_jacobian_norm.std().item(),
                    },
                    used_train_path
                )

                print(
                    f"Saved used training embeddings to: "
                    f"{used_train_path}"
                )

                print(
                    f"Total samples: {len(used_train_embeddings)}"
                )

                print(
                    f"Embedding dimension: "
                    f"{used_train_embeddings.shape[1]}"
                )

                print(
                    f"Overall Jacobian norm: "
                    f"{used_train_jacobian_norm.mean().item():.6f} "
                    f"+/- "
                    f"{used_train_jacobian_norm.std().item():.6f}"
                )


                print('DONE WITH EMBEDDINGS EXTRACTION')
                exit(0)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })
            # print(f"Test env saved in results: {results['args']['test_envs']}")
            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            os.makedirs(os.path.dirname(epochs_path), exist_ok=True)
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            save_accuracy_per_class(os.path.join(args.output_dir, 'out_per_class.txt'), results_per_class)

            std_path = os.path.join(args.output_dir, 'results_std.jsonl')
            os.makedirs(os.path.dirname(std_path), exist_ok=True)
            with open(std_path, 'a') as f:
                f.write(json.dumps(results_std, sort_keys=True) + "\n")

            acc_vs_rd_path = os.path.join(args.output_dir, 'results_vs_random.jsonl')
            os.makedirs(os.path.dirname(acc_vs_rd_path), exist_ok=True)
            with open(acc_vs_rd_path, 'a') as f:
                f.write(json.dumps(results_vs_random, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
            
            # if epoch >= 50: 
            #     break

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')


