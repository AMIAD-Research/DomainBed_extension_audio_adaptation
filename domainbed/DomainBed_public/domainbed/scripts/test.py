# test.py

import argparse
import json
import os
import sys
import time
from sklearn.metrics import f1_score
from pathlib import Path


import numpy as np
import torch
import torchvision
import PIL

from torch.utils.tensorboard import SummaryWriter

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.utils_fr import statistics_splits


SELECTION_METHODS = {
    "iid": model_selection.IIDAccuracySelectionMethod,
    "loo": model_selection.LeaveOneOutSelectionMethod,
    "oracle": model_selection.OracleSelectionMethod,
}


def compute_jacobian_frobenius(logits):

    if logits.ndim != 2:
        raise ValueError(
            "Expected model output with shape [B, C]."
        )

    batch_size, num_classes = logits.shape

    jacobian_squared = torch.zeros(
        batch_size,
        device=x.device,
        dtype=x.dtype,
    )

    for c in range(num_classes):
        grad = torch.autograd.grad(
            outputs=logits[:, c].sum(),
            inputs=x,
            create_graph=False,
            retain_graph=(c < num_classes - 1),
        )[0]

        grad_squared = grad.reshape(batch_size, -1).pow(2).sum(dim=1)

        jacobian_squared += grad_squared

    return torch.sqrt(jacobian_squared)


def collect_embeddings(model, loader, device, domain_id, is_target):
    """
    Extract embeddings and metadata from one domain.

    Returns
    -------
    dict
        {
            "embeddings": Tensor(N,D),
            "labels": Tensor(N),
            "predictions": Tensor(N),
            "domains": Tensor(N),
            "is_target": Tensor(N)
        }
    """

    model.eval()

    embeddings = []
    labels = []
    predictions = []
    jacobian_norm = []
    logits_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            z = model.featurizer(x)

            logits = model.predict(x)
            pred = logits.argmax(dim=1)

            jacobian_frob_norm = compute_jacobian_frobenius(logits=logits)

            embeddings.append(z.cpu())
            labels.append(y.cpu())
            predictions.append(pred.cpu())
            jacobian_norm.append(jacobian_frob_norm.cpu())
            logits_list.append(logits.cpu())

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    jacobian_norm = torch.cat(jacobian_norm)
    logits_list = torch.cat(logits_list)

    domains = torch.full(
        (len(labels),),
        domain_id,
        dtype=torch.long
    )

    is_target = torch.full(
        (len(labels),),
        is_target,
        dtype=torch.bool
    )

    return {
        "embeddings": embeddings,
        "labels": labels,
        "predictions": predictions,
        "domains": domains,
        "is_target": is_target,
        "jacobian_norm": jacobian_norm,
        "logits": logits_list,
    }

def save_configuration_embeddings(
    model,
    dataset,
    test_envs,
    device,
    save_dir,
    record,
):
    """
    Save embeddings and metadata for one source/target configuration.
    Also stores total, domain, and per-class classification accuracies.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    domain_accuracies = {}

    for env_i, env in enumerate(dataset):

        loader = FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS,
        
        )

        data = collect_embeddings(
            model=model,
            loader=loader,
            device=device,
            domain_id=env_i,
            is_target=(env_i in test_envs),
        )

        # Accuracy for this domain
        acc = (data["predictions"] == data["labels"]).float().mean().item()

        domain_accuracies[env_i] = acc
        all_data.append(data)

    # Concaténation globale de toutes les données
    all_embeddings = torch.cat([d["embeddings"] for d in all_data])
    all_labels = torch.cat([d["labels"] for d in all_data])
    all_predictions = torch.cat([d["predictions"] for d in all_data])
    all_domains = torch.cat([d["domains"] for d in all_data])
    all_is_target = torch.cat([d["is_target"] for d in all_data])
    all_jacobian_norm = torch.cat([d["jacobian_norm"] for d in all_data])
    all_logits = torch.cat([d["logits"] for d in all_data])

    # 1. Accuracies globales (Moyenne des domaines vs globale globale)
    source_acc = {
        env: acc
        for env, acc in domain_accuracies.items()
        if env not in test_envs
    }

    target_acc = {
        env: acc for env, acc in domain_accuracies.items() if env in test_envs
    }

    total_global_acc = (
        (all_predictions == all_labels).float().mean().item()
    )

    # 2. Accuracies par classe (tous domaines confondus)
    per_class_acc = {}
    unique_classes = torch.unique(all_labels)

    for cls in unique_classes:
        cls_mask = all_labels == cls
        if cls_mask.sum() > 0:
            per_class_acc[cls.item()] = (
                (all_predictions[cls_mask] == all_labels[cls_mask])
                .float()
                .mean()
                .item()
            )

    # 3. Accuracies croisées : Par Domaine ET Par Classe
    per_domain_per_class_acc = {}
    unique_domains = torch.unique(all_domains)

    for dom in unique_domains:
        dom_mask = all_domains == dom
        per_domain_per_class_acc[dom.item()] = {}

        for cls in unique_classes:
            mask = dom_mask & (all_labels == cls)
            if mask.sum() > 0:
                acc_val = (
                    (all_predictions[mask] == all_labels[mask])
                    .float()
                    .mean()
                    .item()
                )
                per_domain_per_class_acc[dom.item()][cls.item()] = acc_val

    output = {
        "configuration": {
            "test_envs": tuple(test_envs),
            "algorithm": record["args"]["algorithm"]
            if "args" in record
            else None,
            "step": record.get("step"),
        },
        # Embeddings & Métadonnées
        "embeddings": all_embeddings,
        "labels": all_labels,
        "predictions": all_predictions,
        "domains": all_domains,
        "is_target": all_is_target,
        "logits": all_logits,
        "jacobian_norm": all_jacobian_norm,
        # Métriques complètes
        "accuracy": {
            "total_global": total_global_acc,  # Accuracy sur l'ensemble de tous les échantillons
            "per_domain": domain_accuracies,  # Accuracy par domaine (ex: {0: 0.85, 1: 0.82})
            "source_mean": (
                sum(source_acc.values()) / len(source_acc)
                if len(source_acc) > 0
                else None
            ),
            "target_mean": (
                sum(target_acc.values()) / len(target_acc)
                if len(target_acc) > 0
                else None
            ),
            "per_class": per_class_acc,  # Accuracy par classe globale (ex: {0: 0.91, 1: 0.78})
            "per_domain_per_class": per_domain_per_class_acc,  # Dictionnaire croisé {domain_id: {class_id: acc}}
        },
    }

    filename = "target_" + "_".join(map(str, test_envs)) + ".pt"
    torch.save(output, save_dir / filename)

    print(f"Saved embeddings and full accuracies to {save_dir / filename}")

def extract_features(model, loader, device):
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            z = model.featurizer(x)      # or model.featurize([(x, y)])[0]

            features.append(z.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels

# ---------------------------
# Utils
# ---------------------------

def evaluate_with_preds(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model.predict(x).argmax(dim=1)

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred


def evaluate_with_FewShot_preds(model, loader, fewshot_samples, device):
    model.eval()

    with torch.no_grad():
        for x, y in fewshot_samples:
            # print(x)
            # print(y)
            print(f"Number of samples in the fewshot list: {y.shape}")

    
    fewshot_algo = algorithms.FSP(input_shape=model.input_shape, num_classes=4, model_featurizer=model.featurizer, hparams=args.hparams)

    fewshot_algo.compute_prototypes(fewshot_samples)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # preds = model.predict(x).argmax(dim=1)
            preds = fewshot_algo.predict(x)

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    y_pred_labels = np.argmax(y_pred, axis=1)

    

    return y_true, y_pred_labels


def get_fs_samples(args, dataset, device):

    in_splits = []
    out_splits = []
    fewshot_all_splits = [] # To be removed
    fewshot_splits = []

    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*0.2),
            misc.seed_hash(args.seed, env_i))
        
        fewshot_all, in_ = misc.split_dataset(in_,
                int(len(in_)*0.5),
                misc.seed_hash(args.seed, env_i))
        
        fewshot, _ = misc.stratified_split_dataset(fewshot_all,
                int(len(fewshot_all)*args.fs_proportion),
                misc.seed_hash(args.seed, env_i))
        

        in_weights, out_weights, fewshot_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        fewshot_all_splits.append((fewshot_all, fewshot_weights)) 
        fewshot_splits.append((fewshot, fewshot_weights))

    statistics_splits(args.output_dir, in_splits, out_splits, fewshot_all_splits, fewshot_splits)

    fewshot_loaders = []

    for env_i in args.test_envs:
        fewshot_env = fewshot_splits[env_i][0]  # extract dataset only

        loader = FastDataLoader(
            dataset=fewshot_env,
            batch_size=len(fewshot_env),   # full deterministic batch
            num_workers=dataset.N_WORKERS
        )

        fewshot_loaders.append(loader)
    
    fewshot_minibatches_iterator = zip(*fewshot_loaders)
    
    fewshot_device = [(x.to(device), y.to(device))
                    for x,y in next(fewshot_minibatches_iterator)]
    return fewshot_device

def bootstrap_metrics(y_true, y_pred, n_boot=1000):
    n = len(y_true)

    accs = []
    f1s = []

    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)

        yt = y_true[idx]
        yp = y_pred[idx]

        acc = (yt == yp).mean()
        f1 = f1_score(yt, yp, average='macro')

        accs.append(acc)
        f1s.append(f1)

    accs = np.array(accs)
    f1s = np.array(f1s)

    return {
        "acc_mean": accs.mean(),
        "acc_low": np.percentile(accs, 2.5),
        "acc_high": np.percentile(accs, 97.5),

        "f1_mean": f1s.mean(),
        "f1_low": np.percentile(f1s, 2.5),
        "f1_high": np.percentile(f1s, 97.5),
    }


def get_selection_method(name):
    if name == "IIDAccuracy":
        return model_selection.IIDAccuracySelectionMethod
    elif name == "LeaveOneOut":
        return model_selection.LeaveOneOutSelectionMethod
    elif name == "Oracle":
        return model_selection.OracleSelectionMethod
    elif name == "IID_AutoLR_Accuracy":
        return model_selection.IIDAutoLRAccuracySelectionMethod
    else:
        raise NotImplementedError(f"Unknown method: {name}")
    


def sweep_record(selection_method, records):
    hparams_accs = selection_method.hparams_accs(records)

    if not hparams_accs:
        return None
    
    # meilleur hparams (déjà trié)
    _, best_run_records = hparams_accs[0]

    # choisir directement le bon record
    test_records = best_run_records.filter(
        lambda r: len(r['args']['test_envs']) == 1
    )

    if len(test_records) == 1:
        return test_records[0]

    # fallback (rare)
    return test_records[0] if len(test_records) > 0 else None

def sweep_record_run(selection_method, records):
    hparams_accs = selection_method.hparams_accs(records)

    if not hparams_accs:
        return None

    # meilleur hparams (déjà trié)
    accs, best_run_records = hparams_accs[0]

    test_records = best_run_records.filter(
        lambda r: len(r['args']['test_envs']) == 1
    )

    if len(test_records) == 1:
        return accs, test_records[0]

    # fallback (rare)
    if len(test_records) > 0:
        return accs, test_records[0]
    else:
        None, None


def normalize_envs(x):
    if isinstance(x, int):
        return (x,)
    return tuple(x)


def calc_validation_acc(args, record):
    """
    Compute validation accuracy for each validation environment and store
    it using DomainBed's standard metric naming convention:

        env{i}_out_acc

    Example:
        record["env0_out_acc"] = 0.82
        record["env1_out_acc"] = 0.76
        record["env2_out_acc"] = 0.91

    This format is directly compatible with DomainBed model-selection
    methods that access metrics such as:

        record[f"env{test_env}_out_acc"]
    """

    # ---------------------------------------------------------
    # 1. If validation accuracies are already present, don't
    #    recompute them.
    # ---------------------------------------------------------

    validation_hparams = hparams_registry.default_hparams(
        record["args"]["algorithm"],
        args.dataset_validation,
    )

    validation_dataset = vars(datasets)[args.dataset_validation](
        args.data_dir,
        [],
        validation_hparams,
    )

    # Check whether all validation-environment metrics already exist.
    required_keys = [
        f"env{env_i}_out_acc"
        for env_i in range(len(validation_dataset))
    ]

    if all(key in record for key in required_keys):
        return record

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------

    model = load_model(
        record,
        validation_dataset,
        args.device,
    )

    model.eval()

    # ---------------------------------------------------------
    # 3. Evaluate every validation environment
    # ---------------------------------------------------------

    for env_i, env in enumerate(validation_dataset):

        loader = FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=validation_dataset.N_WORKERS,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:

                x = x.to(args.device)
                y = y.to(args.device)

                logits = model.predict(x)
                preds = logits.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        # -----------------------------------------------------
        # DomainBed-compatible metric
        # -----------------------------------------------------

        if total > 0:
            acc = correct / total
        else:
            acc = float("nan")

        record[f"env{env_i}_out_acc"] = acc

        print(
            f"Validation | env{env_i} | "
            f"Acc: {acc:.4f}"
        )

    return record

def load_experiments(args):
    """
    Load experiment records and attach validation accuracies using
    DomainBed's standard metric convention:

        env{i}_out_acc

    Validation accuracies are computed once per experiment and cached in:

        validation_results.jsonl

    The same validation metrics are then attached to every training
    record from that experiment.

    Example record:

        {
            ...
            "env0_out_acc": 0.82,
            "env1_out_acc": 0.76,
            "env2_out_acc": 0.91,
        }
    """

    train_exps_dir = args.train_exps_dir
    records = []
    results_files = []

    # ---------------------------------------------------------
    # Find all results.jsonl files
    # ---------------------------------------------------------

    for entry in os.listdir(train_exps_dir):

        entry_path = os.path.join(train_exps_dir, entry)

        if os.path.isdir(entry_path):

            results_file = os.path.join(
                entry_path,
                "results.jsonl",
            )

            if os.path.isfile(results_file):
                results_files.append(
                    (results_file, entry_path)
                )

        elif entry == "results.jsonl":

            results_files.append(
                (entry_path, train_exps_dir)
            )

    # ---------------------------------------------------------
    # Process each experiment
    # ---------------------------------------------------------

    for results_file, exp_path in results_files:

        validation_cache = os.path.join(
            exp_path,
            "validation_results.jsonl",
        )

        # -----------------------------------------------------
        # Load existing validation cache
        # -----------------------------------------------------

        validation_cache_records = {}

        if os.path.isfile(validation_cache):

            with open(validation_cache, "r") as f:

                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    cached = json.loads(line)

                    step = cached.get("step")

                    validation_cache_records[step] = cached

        # -----------------------------------------------------
        # Load training records
        # -----------------------------------------------------

        with open(results_file, "r") as f:

            experiment_records = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        if not experiment_records:
            continue

        # -----------------------------------------------------
        # Add experiment directory
        # -----------------------------------------------------

        for record in experiment_records:
            record["exp_dir"] = exp_path

        # -----------------------------------------------------
        # Compute validation metrics only for the final model
        # -----------------------------------------------------

        validation_record = experiment_records[-1]

        # Use a fixed cache key because validation is performed
        # once per experiment rather than once per training step.
        cache_key = 1999

        if cache_key in validation_cache_records:

            print(
                f"Validation accuracy already calculated "
                f"for {exp_path}"
            )

            cached = validation_cache_records[cache_key]

        else:

            print(
                f"Calculating validation accuracy "
                f"for {exp_path}"
            )

            validation_record = calc_validation_acc(
                args,
                validation_record,
            )

            # -------------------------------------------------
            # Store only DomainBed-compatible metrics
            # -------------------------------------------------

            cached = {
                "step": validation_record.get("step"),
            }

            for key, value in validation_record.items():

                if (
                    key.startswith("env")
                    and key.endswith("_out_acc")
                ):
                    cached[key] = value

            validation_cache_records[cache_key] = cached

            # -------------------------------------------------
            # Write new validation result to cache
            # -------------------------------------------------

            with open(validation_cache, "a") as f:

                f.write(
                    json.dumps(cached) + "\n"
                )

        # -----------------------------------------------------
        # Attach validation metrics to every training record
        # -----------------------------------------------------

        validation_metrics = {
            key: value
            for key, value in cached.items()
            if (
                key.startswith("env")
                and key.endswith("_out_acc")
            )
        }

        for record in experiment_records:

            record.update(validation_metrics)

            records.append(record)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        f"Loaded {len(records)} records "
        f"from {len(results_files)} experiment(s)"
    )

    return records


def group_by_test_envs(records):
    groups = {}

    for r in records:
        test_envs = normalize_envs(r["args"].get("test_envs", []))

        if test_envs not in groups:
            groups[test_envs] = []

        groups[test_envs].append(r)

    print("\nGrouped experiments:")
    for k, v in groups.items():
        print(f"  test_envs={k}: {len(v)} records")

    return groups


def group_by_run(records):
    groups = {}

    for r in records:

        run_id = r.get("exp_dir", [])
        
        # test_envs = normalize_envs(r["args"].get("test_envs", []))

        if run_id not in groups:
            # print(run_id)
            groups[run_id] = []

        groups[run_id].append(r)

    print("\nGrouped experiments:")
    for k, v in groups.items():
        print(f"  run_id={k}: {len(v)} records")

    return groups

def select_best_per_run(grouped_records, selection_method):

    selection_method = get_selection_method(selection_method)

    best_per_group = {}

    for dir, records_run in grouped_records.items():
        grouped = reporting.get_grouped_records(records_run)

        best_records = []

        for group in grouped:
            accs, best = sweep_record_run(selection_method, group["records"])
            if best is not None:
                best_records.append(best)          

        best_per_group[dir] = (accs, best_records)

        # print(f"\nSelected {len(best_records)} models for dir_exp={dir}")

    return best_per_group


def select_best_per_group_records(grouped_records, selection_method):

    selection_method = get_selection_method(selection_method)

    best_per_group = {}

    for test_envs, records in grouped_records.items():

        grouped = reporting.get_grouped_records(records)

        best_records = []

        for group in grouped:
            best = sweep_record(selection_method, group["records"])
            if best is not None:
                best_records.append(best)

        best_per_group[test_envs] = best_records

        print(f"\nSelected {len(best_records)} models for test_envs={test_envs}")

    return best_per_group

import ast
def load_hparams_from_out_txt(path):
    hparams = {}
    parsing = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("HParams:"):
                parsing = True
                continue

            if parsing:
                if line == "" or ":" not in line:
                    break

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # conversion type robuste
                try:
                    value = ast.literal_eval(value)
                except:
                    # fallback pour True/False ou strings
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    else:
                        try:
                            value = float(value)
                        except:
                            pass

                hparams[key] = value

    return hparams

def load_model(record, dataset, device):

    model_path = os.path.join(record["exp_dir"], "model.pkl")
    out_path = os.path.join(record["exp_dir"], "out.txt")

    print(model_path)
    checkpoint = torch.load(model_path, map_location=device)

    algorithm_class = algorithms.get_algorithm_class(record["args"]["algorithm"])

    # hparams = hparams_registry.random_hparams(record["args"]["algorithm"], record["args"]["dataset"], misc.seed_hash(record["args"]["hparams_seed"], record["args"]["trial_seed"]))
    hparams = load_hparams_from_out_txt(out_path)

    print(f"Num classes: {dataset.num_classes}")

    model = algorithm_class(
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
        num_domains=len(dataset) - len(record["args"]['test_envs']),
        hparams=hparams
    )

    model.load_state_dict(checkpoint["model_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate(model, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model.predict(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# ---------------------------
# Main logic
# ---------------------------

def main(args):
    start_time = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---------------------------
    # Logging
    # ---------------------------
    if args.algorithm != 'FSP': # Evaluation with different modes
        if args.save_features: # Only save features 
            args.output_dir = Path(args.train_exps_dir) / f"new_features_analysis_{args.selection_method_DA}"
        else:
            if args.selection_method_DA == "Oracle": # Oracle selection method - target domain validation set
                args.output_dir = os.path.join(args.train_exps_dir, "test_confidence_oracle")
            elif args.selection_method_DA == "Compare_all_DG": # val vs test performance
                args.output_dir = os.path.join(args.train_exps_dir, "val_vs_test")
            else: # IID source domain validation set
                args.output_dir = os.path.join(args.train_exps_dir, "test_confidence")
    else: # Few Shot approach
        args.output_dir = os.path.join(args.train_exps_dir, args.algorithm, str(args.fs_proportion), "test_confidence")
    os.makedirs(args.output_dir, exist_ok=True)

    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

    print("Environment:")
    print("\tPython:", sys.version.split(" ")[0])
    print("\tPyTorch:", torch.__version__)
    print("\tTorchvision:", torchvision.__version__)
    print("\tCUDA:", torch.version.cuda)
    print("\tCUDNN:", torch.backends.cudnn.version())
    print("\tNumPy:", np.__version__)
    print("\tPIL:", PIL.__version__)

    print("\nArgs:")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")

    # ---------------------------
    # Load dataset
    # ---------------------------
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset_test)

    dataset = vars(datasets)[args.dataset_test](
        args.data_dir,
        [],
        hparams
    )

    args.hparams = hparams
    args.device = device

    # ---------------------------
    # Load experiments
    # ---------------------------
    records = load_experiments(args)

    # ---------------------------
    # Select best per group
    # ---------------------------
    if args.selection_method_DA != "Compare_all_DG":

        # ---------------------------------------------------------
        # 1. Group by test_envs
        # ---------------------------------------------------------
        grouped = group_by_test_envs(records)

        best_per_group = select_best_per_group_records(
            grouped,
            args.selection_method_DA
        )

        # ---------------------------------------------------------
        # 2. Evaluate selected models
        # ---------------------------------------------------------

        final_results = {}
        final_results_source = {}

        for test_envs, records in best_per_group.items():

            print(f"\n=== Evaluating test_envs={test_envs} ===")

            # =========================================================
            # TARGET ENVIRONMENTS
            # =========================================================

            eval_loaders = []
            eval_names = []

            for env_i in test_envs:

                env = dataset[env_i]

                loader = torch.utils.data.DataLoader(
                    env,
                    batch_size=64,
                    shuffle=False,
                    num_workers=dataset.N_WORKERS,
                )

                eval_loaders.append(loader)
                eval_names.append(f"env{env_i}")

            # =========================================================
            # SOURCE ENVIRONMENTS
            # =========================================================

            source_envs = [
                env_i
                for env_i in range(len(dataset))
                if env_i not in test_envs
            ]

            source_loaders = []
            source_names = []

            for env_i in source_envs:

                env = dataset[env_i]

                loader = torch.utils.data.DataLoader(
                    env,
                    batch_size=64,
                    shuffle=False,
                    num_workers=dataset.N_WORKERS,
                )

                source_loaders.append(loader)
                source_names.append(f"env{env_i}")

            # ---------------------------------------------------------
            # Evaluate each selected model
            # ---------------------------------------------------------

            group_results = []
            group_results_source = []

            for i, record in enumerate(records):

                print(f"\nModel {i+1}/{len(records)}")

                model = load_model(
                    record,
                    dataset,
                    device
                )

                # =====================================================
                # TARGET RESULTS
                # =====================================================

                env_scores = {}

                for name, loader in zip(
                    eval_names,
                    eval_loaders
                ):

                    env_i = int(name.replace("env", ""))

                    # -------------------------------------------------
                    # TEST evaluation
                    # -------------------------------------------------

                    if args.algorithm != "v":

                        y_true, y_pred = evaluate_with_preds(
                            model,
                            loader,
                            device
                        )

                    else:

                        args.hparams = record["hparams"]
                        args.test_envs = test_envs

                        fs_dataset = vars(datasets)[args.dataset_FS](
                            args.data_dir,
                            args.test_envs,
                            args.hparams
                        )

                        fewshot_samples = get_fs_samples(
                            args=args,
                            dataset=fs_dataset,
                            device=device
                        )

                        y_true, y_pred = evaluate_with_FewShot_preds(
                            model,
                            loader,
                            fewshot_samples,
                            device
                        )

                    # -------------------------------------------------
                    # Bootstrap test metrics
                    # -------------------------------------------------

                    boot = bootstrap_metrics(
                        y_true,
                        y_pred
                    )

                    env_scores[name] = boot

                    print(
                        f"TARGET {name} | "
                        f"Acc: {boot['acc_mean']:.4f} "
                        f"[{boot['acc_low']:.4f}, "
                        f"{boot['acc_high']:.4f}] | "
                        f"F1: {boot['f1_mean']:.4f} "
                        f"[{boot['f1_low']:.4f}, "
                        f"{boot['f1_high']:.4f}]"
                    )

                group_results.append(env_scores)

                # =====================================================
                # SOURCE RESULTS
                # =====================================================

                env_scores_source = {}

                for name, loader in zip(
                    source_names,
                    source_loaders
                ):

                    env_i = int(name.replace("env", ""))

                    # -------------------------------------------------
                    # SOURCE evaluation
                    # -------------------------------------------------

                    if args.algorithm != "v":

                        y_true, y_pred = evaluate_with_preds(
                            model,
                            loader,
                            device
                        )

                    else:

                        args.hparams = record["hparams"]
                        args.test_envs = test_envs

                        fs_dataset = vars(datasets)[args.dataset_FS](
                            args.data_dir,
                            args.test_envs,
                            args.hparams
                        )

                        fewshot_samples = get_fs_samples(
                            args=args,
                            dataset=fs_dataset,
                            device=device
                        )

                        y_true, y_pred = evaluate_with_FewShot_preds(
                            model,
                            loader,
                            fewshot_samples,
                            device
                        )

                    # -------------------------------------------------
                    # Bootstrap source metrics
                    # -------------------------------------------------

                    boot = bootstrap_metrics(
                        y_true,
                        y_pred
                    )

                    env_scores_source[name] = boot

                    print(
                        f"SOURCE {name} | "
                        f"Acc: {boot['acc_mean']:.4f} "
                        f"[{boot['acc_low']:.4f}, "
                        f"{boot['acc_high']:.4f}] | "
                        f"F1: {boot['f1_mean']:.4f} "
                        f"[{boot['f1_low']:.4f}, "
                        f"{boot['f1_high']:.4f}]"
                    )

                group_results_source.append(env_scores_source)

            # =========================================================
            # Aggregate TARGET results
            # =========================================================

            agg = {}

            for env in eval_names:

                acc_means = [
                    r[env]["acc_mean"]
                    for r in group_results
                ]

                acc_lows = [
                    r[env]["acc_low"]
                    for r in group_results
                ]

                acc_highs = [
                    r[env]["acc_high"]
                    for r in group_results
                ]

                f1_means = [
                    r[env]["f1_mean"]
                    for r in group_results
                ]

                f1_lows = [
                    r[env]["f1_low"]
                    for r in group_results
                ]

                f1_highs = [
                    r[env]["f1_high"]
                    for r in group_results
                ]

                agg[env] = {
                    "acc_mean": np.mean(acc_means),
                    "acc_low": np.mean(acc_lows),
                    "acc_high": np.mean(acc_highs),

                    "f1_mean": np.mean(f1_means),
                    "f1_low": np.mean(f1_lows),
                    "f1_high": np.mean(f1_highs),
                }

            final_results[str(test_envs)] = agg

            # =========================================================
            # Aggregate SOURCE results
            # =========================================================

            agg_source = {}

            for env in source_names:

                acc_means = [
                    r[env]["acc_mean"]
                    for r in group_results_source
                ]

                acc_lows = [
                    r[env]["acc_low"]
                    for r in group_results_source
                ]

                acc_highs = [
                    r[env]["acc_high"]
                    for r in group_results_source
                ]

                f1_means = [
                    r[env]["f1_mean"]
                    for r in group_results_source
                ]

                f1_lows = [
                    r[env]["f1_low"]
                    for r in group_results_source
                ]

                f1_highs = [
                    r[env]["f1_high"]
                    for r in group_results_source
                ]

                agg_source[env] = {
                    "acc_mean": np.mean(acc_means),
                    "acc_low": np.mean(acc_lows),
                    "acc_high": np.mean(acc_highs),

                    "f1_mean": np.mean(f1_means),
                    "f1_low": np.mean(f1_lows),
                    "f1_high": np.mean(f1_highs),
                }

            final_results_source[str(test_envs)] = agg_source

            # =========================================================
            # Print TARGET results
            # =========================================================

            print(
                f"\nAggregated TARGET results for "
                f"test_envs={test_envs}:"
            )

            for env, res in agg.items():

                print(
                    f"{env} | "
                    f"Acc: {res['acc_mean']:.4f} "
                    f"[{res['acc_low']:.4f}, "
                    f"{res['acc_high']:.4f}] | "
                    f"F1: {res['f1_mean']:.4f} "
                    f"[{res['f1_low']:.4f}, "
                    f"{res['f1_high']:.4f}]"
                )

            # =========================================================
            # Print SOURCE results
            # =========================================================

            print(
                f"\nAggregated SOURCE results for "
                f"test_envs={test_envs}:"
            )

            for env, res in agg_source.items():

                print(
                    f"{env} | "
                    f"Acc: {res['acc_mean']:.4f} "
                    f"[{res['acc_low']:.4f}, "
                    f"{res['acc_high']:.4f}] | "
                    f"F1: {res['f1_mean']:.4f} "
                    f"[{res['f1_low']:.4f}, "
                    f"{res['f1_high']:.4f}]"
                )
    else:
        # ---------------------------
        # Evaluate all
        # ---------------------------

        final_results = {}

        # group by run IDs
        run_groups = group_by_run(records)
        # Select the step from each run based on target domain validation set
        best_per_group = select_best_per_run(
            run_groups,
            "Oracle"
        )

        # ---------------------------
        # Loop over ALL experiments
        # ---------------------------

        for i, record in enumerate(best_per_group):

            record = best_per_group[record]
            accs, record = record     
            record = record[0] 

            exp_id = record.get("exp_dir").split('/')[-1]
            test_env = normalize_envs(record["args"].get("test_envs", []))[0]
            print(f"EXP {exp_id} on test_env {test_env}")

            print(f"\nModel {i+1}/{len(best_per_group)} | ID: {exp_id}")

            model = load_model(record, dataset, device)

            result_row = {
                "exp_id": exp_id,
                "test_envs": test_env,
            }

            env = dataset[test_env]
            loader = FastDataLoader(
                dataset=env,
                batch_size=64,
                num_workers=dataset.N_WORKERS,
            
            )

            # ---------------------------
            # Get OUT (validation) score
            # ---------------------------

            result_row[f"env{test_env}_val_out_acc"] = accs['val_acc']
            # print(result_row["out_acc"])

            # ---------------------------
            # Evaluate on test envs
            # ---------------------------

            y_true, y_pred = evaluate_with_preds(model, loader, device)
            boot = bootstrap_metrics(y_true, y_pred)

            result_row[f"env{test_env}_test_acc_mean"] = boot["acc_mean"]
            result_row[f"env{test_env}_test_acc_low"]  = boot["acc_low"]
            result_row[f"env{test_env}_test_acc_high"] = boot["acc_high"]

            print(result_row)

            final_results[str(exp_id)] = result_row
                    
    # ---------------------------
    # Save results
    # ---------------------------
    results_path = os.path.join(args.output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    results_path_source = os.path.join(args.output_dir, "final_results_source.json")
    with open(results_path_source, "w") as f:
        json.dump(final_results_source, f, indent=2)

    print(f"\nSaved target results to {results_path} and source results to {results_path_source}")

    elapsed = time.monotonic() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_test", type=str, required=True)
    parser.add_argument("--dataset_validation", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--save_features", default=False)

    parser.add_argument("--selection_method_DA", type=str, default="IIDAccuracy", choices=["IIDAccuracy", "Oracle"])
    parser.add_argument("--train_exps_dir", type=str, required=True)   
    parser.add_argument("--modality", type=str, default='audio', choices=['audio', 'image'])   

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fs_proportion", type=float, default=0)
    parser.add_argument('--dataset_FS', type=str, default="NOISY_AUDIO_DA_CWWS")


    args = parser.parse_args()

    main(args)