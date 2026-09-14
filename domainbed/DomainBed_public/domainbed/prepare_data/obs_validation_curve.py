
## Plot des performances
# uv run --no-sync python -m domainbed.prepare_data.obs_validation_curve

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import re

from torch.utils.data import DataLoader

from domainbed import algorithms
from domainbed import datasets
from domainbed.scripts.test import load_hparams_from_out_txt


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Validation curve trained models")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

parser.add_argument(
    "--data_dir",
    type=str,
    default="/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="noisy_CWWS_validation"
)

parser.add_argument(
    "--json_path",
    type=str,
    default="domainbed/prepare_data/noisy_CWWS_validation.json"
)

args = parser.parse_args()


# Mono_clean 
exp_path = Path(
    "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono/7aed3d85e9867729c6ac64683221996a_bis/"
)


out_path = exp_path / "out.txt"


checkpoint_paths = sorted(
    exp_path.glob("model_step*.pkl"),
    key=lambda p: int(
        re.search(r"model_step(\d+)\.pkl", p.name).group(1)
    )
)

if len(checkpoint_paths) == 0:
    raise FileNotFoundError(
        f"Aucun checkpoint model_step*.pkl trouvé dans {exp_path}"
    )

print("\nCheckpoints trouvés :")
for path in checkpoint_paths:
    print("  ", path.name)


# ============================================================
# HYPERPARAMETRES / DATASET
# ============================================================

hparams = load_hparams_from_out_txt(out_path)

dataset_class = vars(datasets)[args.dataset]

dataset = dataset_class(
    args.data_dir,
    [],
    hparams
)

print("\nDataset")
print("Input shape:", dataset.input_shape)
print("Num classes:", dataset.num_classes)
print("Num domains:", len(dataset.datasets))


# ============================================================
# DOMAINES
# ============================================================

domain_names = [
    "clean",
    "saturation",
    "wham",
    "wind",
]


# ============================================================
# FONCTION D'EVALUATION D'UN CHECKPOINT
# ============================================================

def evaluate_checkpoint(model_path):

    print("\n" + "=" * 70)
    print(f"EVALUATION : {model_path.name}")
    print("=" * 70)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model_hparams = checkpoint["model_hparams"]

    algorithm_class = algorithms.get_algorithm_class(
        checkpoint["args"]["algorithm"]
    )

    model = algorithm_class(
        input_shape=checkpoint["model_input_shape"],
        num_classes=checkpoint["model_num_classes"],
        num_domains=checkpoint["model_num_domains"],
        hparams=model_hparams
    )

    model.load_state_dict(
        checkpoint["model_dict"]
    )

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    # Accuracy de chaque domaine
    domain_accuracies = {}

    with torch.no_grad():

        for domain_idx, domain_name in enumerate(domain_names):

            print(f"\nDOMAIN : {domain_name}")

            env_dataset = dataset.datasets[domain_idx]

            loader = DataLoader(
                env_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=(device.type == "cuda")
            )

            domain_correct = 0
            domain_total = 0

            for batch in loader:

                x, y = batch

                x = x.to(device)
                y = y.to(device)

                logits = model.predict(x)
                predictions = logits.argmax(dim=1)

                correct = (
                    predictions == y
                )

                domain_correct += correct.sum().item()
                domain_total += len(y)

            # --------------------------------------------
            # Accuracy domaine
            # --------------------------------------------

            accuracy = (
                domain_correct / domain_total
                if domain_total > 0
                else 0.0
            )

            domain_accuracies[domain_name] = accuracy

            total_correct += domain_correct
            total_samples += domain_total

            print(
                f"Accuracy {domain_name}: "
                f"{accuracy:.4f} "
                f"({domain_correct}/{domain_total})"
            )

    # --------------------------------------------------------
    # Accuracy globale
    # --------------------------------------------------------

    global_accuracy = (
        total_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    print(
        f"\nAccuracy globale : "
        f"{global_accuracy:.4f} "
        f"({total_correct}/{total_samples})"
    )

    return global_accuracy, domain_accuracies


# ============================================================
# EVALUATION DE TOUS LES CHECKPOINTS
# ============================================================

validation_steps = []
validation_accuracies = []

# Une liste d'accuracy par domaine
validation_domain_accuracies = {
    domain: []
    for domain in domain_names
}


for model_path in checkpoint_paths:

    match = re.search(
        r"model_step(\d+)\.pkl",
        model_path.name
    )

    if match is None:
        continue

    step = int(match.group(1))

    # --------------------------------------------------------
    # Evaluation du checkpoint
    # --------------------------------------------------------

    global_accuracy, domain_accuracies = (
        evaluate_checkpoint(model_path)
    )

    validation_steps.append(step)
    validation_accuracies.append(
        global_accuracy
    )

    # --------------------------------------------------------
    # Sauvegarde accuracy par domaine
    # --------------------------------------------------------

    for domain in domain_names:

        validation_domain_accuracies[
            domain
        ].append(
            domain_accuracies[domain]
        )


# ============================================================
# TRI PAR STEP
# ============================================================

sorted_indices = np.argsort(
    validation_steps
)

validation_steps = [
    validation_steps[i]
    for i in sorted_indices
]

validation_accuracies = [
    validation_accuracies[i]
    for i in sorted_indices
]

for domain in domain_names:

    validation_domain_accuracies[domain] = [
        validation_domain_accuracies[domain][i]
        for i in sorted_indices
    ]


# ============================================================
# SAUVEGARDE DES RESULTATS DE VALIDATION
# ============================================================

validation_results_path = (
    exp_path /
    f"validation_curve_{args.dataset}.json"
)

validation_results = [
    {
        "step": int(step),
        "accuracy": float(accuracy)
    }
    for step, accuracy in zip(
        validation_steps,
        validation_accuracies
    )
]

with open(validation_results_path, "w") as f:

    json.dump(
        validation_results,
        f,
        indent=2
    )

print(
    f"\nRésultats validation sauvegardés dans : "
    f"{validation_results_path}"
)


# ============================================================
# PLOT VALIDATION CURVE
# ============================================================

validation_figure_path = (
    exp_path /
    f"validation_curve_{args.dataset}.png"
)

fig, axes = plt.subplots(
    5,
    1,
    figsize=(10, 20),
    sharex=True
)

# ------------------------------------------------------------
# 1. Accuracy globale
# ------------------------------------------------------------

axes[0].plot(
    validation_steps,
    validation_accuracies,
    marker="o",
    linewidth=2
)

axes[0].set_title(
    "Global validation accuracy",
    fontsize=14
)

axes[0].set_ylabel(
    "Accuracy",
    fontsize=12
)

axes[0].set_ylim(
    0,
    1.05
)

axes[0].grid(
    True,
    alpha=0.3
)


# ------------------------------------------------------------
# 2-5. Accuracy par domaine
# ------------------------------------------------------------

for ax, domain in zip(
    axes[1:],
    domain_names
):

    ax.plot(
        validation_steps,
        validation_domain_accuracies[domain],
        marker="o",
        linewidth=2
    )

    ax.set_title(
        f"{domain.capitalize()} validation accuracy",
        fontsize=14
    )

    ax.set_ylabel(
        "Accuracy",
        fontsize=12
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        True,
        alpha=0.3
    )


# ------------------------------------------------------------
# Axe X commun
# ------------------------------------------------------------

axes[-1].set_xlabel(
    "Training step",
    fontsize=14
)


# ------------------------------------------------------------
# Titre global
# ------------------------------------------------------------

fig.suptitle(
    "Validation accuracy as a function of training step",
    fontsize=17,
    y=0.995
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

plt.savefig(
    validation_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nValidation curve sauvegardée dans : "
    f"{validation_figure_path}"
)