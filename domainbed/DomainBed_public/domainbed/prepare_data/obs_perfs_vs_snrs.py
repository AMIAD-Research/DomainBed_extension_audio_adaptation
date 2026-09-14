# Attention, on a bien laissé l'option mono source dans le train.py de domainbed 

## Entrainement Mono-source 
# uv run --no-sync python -m domainbed.scripts.train --algorithm ERM --data_dir /lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/ --dataset noisy_CWWS_train_mono_wide --holdout_fraction 0.2 --hparams_seed 1 --output_dir /lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono_discover_snr/clean --seed 2040398665 --selection_method_DA IIDAcuuracy --task domain_generalization --test_envs 0 --trial_seed 0 --uda_holdout_fraction 0

## Plot des performances
# uv run --no-sync python -m domainbed.prepare_data.obs_perfs_vs_snrs

import argparse
import json
from pathlib import Path
from collections import defaultdict
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from domainbed import algorithms
from domainbed import datasets
from domainbed.scripts.test import load_hparams_from_out_txt


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Evaluate trained models")

parser.add_argument(
    "--data_dir",
    type=str,
    default="/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="noisy_CWWS_test"
)

parser.add_argument(
    "--json_path",
    type=str,
    default="domainbed/prepare_data/noisy_CWWS_test.json"
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

### DG
# Mauvais sur clean - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/DG/erm/791e433371cf61045baaf9cdec670c53"
# )

# Mauvais sur saturation - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/DG/erm/26b8df5e5b6d0f6f1e5fca95735945c0"
# )

# Mauvais sur wham - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/DG/erm/fa7ef595213c4be462b594a4787f6263"
# )

# Mauvais sur wind - 
exp_path = Path(
    "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/DG/erm/cea58d956c4fab21488d90d3e4ae159a"
)

### Mono

# mono sur clean - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono/27532667c4a342bf6e3052edd8e88ec0"
# )

# mono sur saturation - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono/089750a196e2c3a06034719915df3317"
# )

# mono sur wham - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono/bc9c46aa8d3033532fa8e78ec2bfbba6"
# )

# mono sur wind - 
# exp_path = Path(
#     "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/results/CWWS/mono/9ae7240b77b5e499442c760848815734"
# )



print(f"EXP PATH: {exp_path}")

# ============================================================
# PATHS
# ============================================================

out_path = exp_path / "out.txt"
model_path = exp_path / "model.pkl"

results_path = (
    exp_path
    / f"evaluation_results_{args.dataset}.json"
)

figure_path = (
    exp_path
    / f"accuracy_vs_snr_gain_{args.dataset}.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model:", model_path)


# ============================================================
# CHARGEMENT DU JSON
# ============================================================

with open(args.json_path, "r") as f:
    json_data = json.load(f)

print(
    f"Nombre d'entrées dans le JSON : "
    f"{len(json_data['data'])}"
)


# ============================================================
# IDENTIFICATION DU DOMAINE
# ============================================================

def get_domain(item):
    """
    Détermine le domaine à partir des métadonnées du JSON.

    A adapter si les conventions utilisées dans ton JSON
    sont différentes.
    """

    noise = item.get("noise")
    gain = item.get("gain")
    gustiness = item.get("gustiness")
    wind_profile = item.get("wind_profile")

    # ----------------------------
    # CLEAN
    # ----------------------------

    if noise == "clean":
        return "clean"

    # ----------------------------
    # WHAM
    # ----------------------------

    if noise is not None:

        noise_lower = str(noise).lower()

        if "wham" in noise_lower:
            return "wham"

    # ----------------------------
    # WIND
    # ----------------------------

    if (
        gustiness is not None
        or wind_profile is not None
    ):
        return "wind"

    if noise is not None:

        noise_lower = str(noise).lower()

        if "wind" in noise_lower:
            return "wind"

    # ----------------------------
    # SATURATION
    # ----------------------------

    if gain is not None:
        return "saturation"

    raise ValueError(
        "Impossible de déterminer le domaine pour :\n"
        f"{item}"
    )


# ============================================================
# CONSTRUCTION DE L'INDEX DES METADONNEES
# ============================================================

metadata_by_path = {}
metadata_by_filename = defaultdict(list)


for item in json_data["data"]:

    clean_path = item["clean"]

    domain = get_domain(item)

    info = {
        "domain": domain,
        "clean_path": clean_path,

        "snr_db": item.get("snr_db"),
        "gain": item.get("gain"),
        "gustiness": item.get("gustiness"),
        "wind_profile": item.get("wind_profile"),

        "noise": item.get("noise"),
    }

    # Clé principale : chemin relatif
    metadata_by_path[clean_path] = info

    # Clé secondaire : nom du fichier
    filename = Path(clean_path).name

    metadata_by_filename[filename].append(info)


print("\nNombre d'entrées par domaine dans le JSON:")

domain_counts = defaultdict(int)

for info in metadata_by_path.values():

    domain_counts[info["domain"]] += 1


for domain, count in sorted(domain_counts.items()):

    print(
        f"  {domain}: {count}"
    )


# ============================================================
# CHARGEMENT DU CHECKPOINT
# ============================================================

checkpoint = torch.load(
    model_path,
    map_location=device
)


# ============================================================
# HYPERPARAMETRES
# ============================================================

hparams = load_hparams_from_out_txt(
    out_path
)


# ============================================================
# DATASET
# ============================================================

dataset_class = vars(datasets)[args.dataset]

dataset = dataset_class(
    args.data_dir,
    [],
    hparams
)

print("\nDataset")
print(
    "Input shape:",
    dataset.input_shape
)

print(
    "Num classes:",
    dataset.num_classes
)

print(
    "Num domains:",
    len(dataset.datasets)
)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

checkpoint = torch.load(
    model_path,
    map_location=device
)

hparams = checkpoint["model_hparams"]

algorithm_class = algorithms.get_algorithm_class(
    checkpoint["args"]["algorithm"]
)

model = algorithm_class(
    input_shape=checkpoint["model_input_shape"],
    num_classes=checkpoint["model_num_classes"],
    num_domains=checkpoint["model_num_domains"],
    hparams=hparams
)

model.load_state_dict(
    checkpoint["model_dict"]
)

model.to(device)
model.eval()


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
# RESULTATS
# ============================================================

results = {}

for domain in domain_names:

    results[domain] = {}

    for class_id in range(
        dataset.num_classes
    ):

        results[domain][str(class_id)] = {}


# ============================================================
# STATISTIQUES POUR LES PLOTS
# ============================================================

snr_values = defaultdict(list)

gain_values = defaultdict(list)


# ============================================================
# FONCTION POUR TROUVER LES METADONNEES
# ============================================================

def find_metadata(filename, domain):
    """
    Recherche les métadonnées correspondant à un fichier.

    On essaye d'abord une correspondance exacte sur le nom.
    """

    filename = Path(filename).name

    candidates = metadata_by_filename.get(
        filename,
        []
    )

    candidates = [
        x
        for x in candidates
        if x["domain"] == domain
    ]

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) > 1:

        print(
            f"WARNING: plusieurs métadonnées pour "
            f"{filename} dans {domain}"
        )

        return candidates[0]

    return None


# ============================================================
# EVALUATION
# ============================================================

total_correct = 0
total_samples = 0


model.eval()

with torch.no_grad():

    for domain_idx, domain_name in enumerate(
        domain_names
    ):

        print("\n")
        print("=" * 70)
        print(
            f"DOMAIN : {domain_name}"
        )
        print("=" * 70)

        env_dataset = dataset.datasets[
            domain_idx
        ]

        # ----------------------------------------
        # Récupération des chemins
        # ----------------------------------------

        if hasattr(
            env_dataset,
            "samples"
        ):

            samples = env_dataset.samples

        elif hasattr(
            env_dataset,
            "imgs"
        ):

            samples = env_dataset.imgs

        else:

            raise AttributeError(
                f"Le dataset {domain_name} ne possède "
                f"ni 'samples' ni 'imgs'."
            )

        # ----------------------------------------
        # DataLoader
        # ----------------------------------------

        loader = DataLoader(
            env_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=(
                device.type == "cuda"
            )
        )

        domain_correct = 0
        domain_total = 0

        # ----------------------------------------
        # Batches
        # ----------------------------------------

        for batch_idx, batch in enumerate(
            loader
        ):

            x, y = batch

            x = x.to(device)
            y = y.to(device)

            logits = model.predict(x)

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------
            # Parcours des exemples
            # ------------------------------------

            start_idx = (
                batch_idx
                * loader.batch_size
            )

            for i in range(
                len(y)
            ):

                sample_idx = (
                    start_idx + i
                )

                filename, dataset_label = (
                    samples[sample_idx]
                )

                filename = Path(
                    filename
                ).name

                true_label = int(
                    y[i].item()
                )

                predicted_label = int(
                    predictions[i].item()
                )

                correct = (
                    predicted_label
                    == true_label
                )

                # --------------------------------
                # Métadonnées
                # --------------------------------

                info = find_metadata(
                    filename,
                    domain_name
                )

                if info is None:

                    print(
                        "WARNING: métadonnées "
                        f"introuvables pour "
                        f"{filename} "
                        f"({domain_name})"
                    )

                    snr_db = None
                    gain = None
                    gustiness = None
                    wind_profile = None
                    noise = None
                    clean_path = None

                else:

                    snr_db = info["snr_db"]

                    gain = info["gain"]

                    gustiness = info[
                        "gustiness"
                    ]

                    wind_profile = info[
                        "wind_profile"
                    ]

                    noise = info["noise"]

                    clean_path = info[
                        "clean_path"
                    ]

                # --------------------------------
                # Enregistrement résultat
                # --------------------------------

                results[
                    domain_name
                ][
                    str(true_label)
                ][
                    filename
                ] = {

                    "correct": bool(
                        correct
                    ),

                    "true_label": (
                        true_label
                    ),

                    "predicted_label": (
                        predicted_label
                    ),

                    "snr_db": snr_db,

                    "gain": gain,

                    "gustiness": (
                        gustiness
                    ),

                    "wind_profile": (
                        wind_profile
                    ),

                    "noise": noise,

                    "clean_path": (
                        clean_path
                    ),
                }

                # --------------------------------
                # Statistiques SNR
                # --------------------------------

                if snr_db is not None:

                    snr_values[
                        domain_name
                    ].append(
                        {
                            "snr": float(
                                snr_db
                            ),

                            "correct": bool(
                                correct
                            )
                        }
                    )

                # --------------------------------
                # Statistiques Gain
                # --------------------------------

                if gain is not None:

                    gain_values[
                        domain_name
                    ].append(
                        {
                            "gain": float(
                                gain
                            ),

                            "correct": bool(
                                correct
                            )
                        }
                    )

                # --------------------------------
                # Compteurs
                # --------------------------------

                domain_correct += int(
                    correct
                )

                domain_total += 1

        # ----------------------------------------
        # Accuracy domaine
        # ----------------------------------------

        if domain_total > 0:

            accuracy = (
                domain_correct
                / domain_total
            )

        else:

            accuracy = 0.0

        total_correct += domain_correct

        total_samples += domain_total

        print(
            f"\nAccuracy {domain_name}: "
            f"{accuracy:.4f} "
            f"({domain_correct}/{domain_total})"
        )


# ============================================================
# ACCURACY GLOBALE
# ============================================================

global_accuracy = (
    total_correct
    / total_samples
)


print("\n")
print("=" * 70)
print("GLOBAL")
print("=" * 70)

print(
    f"Accuracy globale : "
    f"{global_accuracy:.4f} "
    f"({total_correct}/{total_samples})"
)


# ============================================================
# ACCURACY PAR CLASSE
# ============================================================

print("\n")
print("=" * 70)
print("ACCURACY PAR DOMAINE ET PAR CLASSE")
print("=" * 70)


for domain in domain_names:

    print(f"\n{domain}")

    for class_id in range(
        dataset.num_classes
    ):

        class_results = results[
            domain
        ][
            str(class_id)
        ]

        if len(class_results) == 0:

            print(
                f"  Classe {class_id}: "
                "aucune donnée"
            )

            continue

        correct = sum(
            x["correct"]
            for x in class_results.values()
        )

        total = len(
            class_results
        )

        accuracy = (
            correct / total
        )

        print(
            f"  Classe {class_id}: "
            f"{accuracy:.4f} "
            f"({correct}/{total})"
        )


# ============================================================
# SAUVEGARDE DES RESULTATS
# ============================================================

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    f"\nRésultats sauvegardés dans : "
    f"{results_path}"
)


# ============================================================
# CREATION DE LA FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)


# ============================================================
# SUBPLOT 1
# ACCURACY VS SNR
# ============================================================

ax = axes[0]


for domain in domain_names:

    data = snr_values[
        domain
    ]

    # --------------------------------------------------------
    # Domaine sans SNR
    # --------------------------------------------------------

    if len(data) == 0:

        print(
            f"\nPas de SNR pour {domain}, "
            "domaine ignoré dans le plot."
        )

        continue

    snrs = np.array([
        x["snr"]
        for x in data
    ])

    correct = np.array([
        x["correct"]
        for x in data
    ])

    # --------------------------------------------------------
    # Création des bins SNR
    # --------------------------------------------------------

    if snrs.min() == snrs.max():

        bin_centers = [
            snrs.min()
        ]

        accuracies = [
            correct.mean()
        ]

    else:

        bins = np.linspace(
            snrs.min(),
            snrs.max(),
            11
        )

        bin_centers = []
        accuracies = []

        for i in range(
            len(bins) - 1
        ):

            if i == len(bins) - 2:

                mask = (
                    (snrs >= bins[i])
                    &
                    (snrs <= bins[i + 1])
                )

            else:

                mask = (
                    (snrs >= bins[i])
                    &
                    (snrs < bins[i + 1])
                )

            if mask.sum() == 0:
                continue

            bin_center = (
                bins[i]
                + bins[i + 1]
            ) / 2

            accuracy = (
                correct[mask].mean()
            )

            bin_centers.append(
                bin_center
            )

            accuracies.append(
                accuracy
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.plot(
        bin_centers,
        accuracies,
        marker="o",
        linewidth=2,
        label=domain
    )


ax.set_xlabel(
    "SNR (dB)",
    fontsize=14
)

ax.set_ylabel(
    "Accuracy",
    fontsize=14
)

ax.set_title(
    "Classification accuracy vs SNR",
    fontsize=15
)

ax.set_ylim(
    0,
    1.05
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


# ============================================================
# SUBPLOT 2
# ACCURACY VS GAIN
# ============================================================

ax = axes[1]


domain = "saturation"

data = gain_values[
    domain
]


if len(data) == 0:

    print(
        "\nPas de gain pour le domaine "
        "saturation."
    )

else:

    gains = np.array([
        x["gain"]
        for x in data
    ])

    correct = np.array([
        x["correct"]
        for x in data
    ])

    # --------------------------------------------------------
    # Création des bins Gain
    # --------------------------------------------------------

    if gains.min() == gains.max():

        bin_centers = [
            gains.min()
        ]

        accuracies = [
            correct.mean()
        ]

    else:

        bins = np.linspace(
            gains.min(),
            gains.max(),
            11
        )

        bin_centers = []
        accuracies = []

        for i in range(
            len(bins) - 1
        ):

            if i == len(bins) - 2:

                mask = (
                    (gains >= bins[i])
                    &
                    (gains <= bins[i + 1])
                )

            else:

                mask = (
                    (gains >= bins[i])
                    &
                    (gains < bins[i + 1])
                )

            if mask.sum() == 0:
                continue

            bin_center = (
                bins[i]
                + bins[i + 1]
            ) / 2

            accuracy = (
                correct[mask].mean()
            )

            bin_centers.append(
                bin_center
            )

            accuracies.append(
                accuracy
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.plot(
        bin_centers,
        accuracies,
        marker="o",
        linewidth=2,
        label="saturation"
    )


ax.set_xlabel(
    "Gain",
    fontsize=14
)

ax.set_ylabel(
    "Accuracy",
    fontsize=14
)

ax.set_title(
    "Classification accuracy vs gain",
    fontsize=15
)

ax.set_ylim(
    0,
    1.05
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()


# ============================================================
# STYLE GLOBAL DE LA FIGURE
# ============================================================

fig.suptitle(
    "Classification performance",
    fontsize=17
)

fig.tight_layout()


# ============================================================
# SAUVEGARDE
# ============================================================

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    f"\nFigure sauvegardée dans : "
    f"{figure_path}"
)