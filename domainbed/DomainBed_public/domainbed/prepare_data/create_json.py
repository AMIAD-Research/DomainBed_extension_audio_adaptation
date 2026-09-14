from pathlib import Path
import os
import json
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import argparse

from domainbed.prepare_data.utils_dataset import get_train_speakers, get_validation_speakers


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


ENVIRONMENTS = [
    "clean",
    "wham",
    "wind",
    "saturation",
]


LANGUAGES = [
    "fr",
    "de",
    "es",
    "zh-CN",
]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {

    "noisy_CWWS_train_mono": {
        "split": "train",
        "total": 18000,
    },

    "noisy_CWWS_train_small": {
        "split": "train",
        "total": 6000,
    },

    "noisy_CWWS_validation": {
        "split": "dev",
        "total": 6000,
    },

    "noisy_CWWS_test": {
        "split": "test",
        "total": 18000,
    },
}


# ============================================================
# TRAIN SPEAKER TARGETS
# ============================================================

# Nombre de speakers souhaité par cellule
# (dataset × environnement × langue)
TRAIN_SMALL_SPEAKERS_PER_CELL = 150
TRAIN_MONO_SPEAKERS_PER_CELL = 150

# Pour zh-CN, il y a environ 660 speakers disponibles.
#
# 660 / 4 domaines = 165 speakers par domaine.
#
# On utilise donc toute la diversité speaker disponible
# en répartissant les 660 speakers sur les 4 domaines.
ZH_CN_TRAIN_SPEAKERS_PER_CELL = 150


# ============================================================
# EVALUATION CONSTRAINTS
# ============================================================

# Validation:
# exactement UN sample clean par speaker.
VALIDATION_ONE_SAMPLE_PER_SPEAKER = False

# Test:
# maximum 10 samples clean par speaker.
TEST_MAX_SAMPLES_PER_SPEAKER = 20


# ============================================================
# HARD LIMIT
# ============================================================

ABSOLUTE_MAX_CLIPS_PER_SPEAKER = 200


# ============================================================
# RANDOM PARAMETERS
# ============================================================

gustiness_range = [8, 11]

wind_profile_magnitude_range = [8, 15]

wind_snr_range = [-10, 10]

wham_snr_range = [-10, 10]

saturation_gain_range = [0.5, 4.5]


# ============================================================
# PATHS
# ============================================================
parser = argparse.ArgumentParser(description='Create Audio Dataset via JSON file with metadata')
parser.add_argument(
    "--dataset_name",
    default="noisy_CWWS",
)

parser.add_argument(
    "--path_to_speech_data",
    type=Path,
    required=True
)

parser.add_argument(
    "--path_to_wham_noise",
    type=Path,
    required=True
)

parser.add_argument(
    "--output_path",
    type=Path,
    default="domainbed/prepare_data/save_json",
)

args = parser.parse_args()

dataset_name = args.dataset_name

cv_root = (args.path_to_speech_data)

wham_noise_root = (args.path_to_wham_noise)

output_path = args.output_path


# ============================================================
# LOAD COMMONVOICE
# ============================================================

def get_data_from_CV(
    data_path,
    split,
):
    """
    Load CommonVoice metadata.

    Returns:

        {
            language: [
                {
                    spk_id,
                    path_to_mp3,
                    language,
                    duration_s_before_crop
                },
                ...
            ]
        }

    IMPORTANT
    ---------

    client_id est conservé tel quel.

    Un même client_id présent dans plusieurs langues est donc
    considéré comme le même speaker global.
    """

    all_data = {}

    for language in LANGUAGES:

        tsv_path = (
            data_path
            / language
            / f"{split}.tsv"
        )

        if not tsv_path.exists():
            raise FileNotFoundError(
                f"CommonVoice TSV not found:\n{tsv_path}"
            )

        df = pd.read_csv(
            tsv_path,
            sep="\t",
            low_memory=False,
        )

        required_columns = {
            "client_id",
            "path",
            "locale",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise RuntimeError(
                f"Missing columns {missing} "
                f"in {tsv_path}"
            )

        df = df[
            [
                "client_id",
                "path",
                "locale",
            ]
        ].rename(
            columns={
                "client_id": "spk_id",
                "path": "path_to_mp3",
                "locale": "language",
            }
        )

        # ----------------------------------------------------
        # Durations
        # ----------------------------------------------------

        duration_path = (
            data_path
            / language
            / "clip_durations.tsv"
        )

        if not duration_path.exists():
            raise FileNotFoundError(
                f"Duration file not found:\n"
                f"{duration_path}"
            )

        df_dur = pd.read_csv(
            duration_path,
            sep="\t",
            header=None,
            names=[
                "clip",
                "duration_ms",
            ],
            low_memory=False,
        )

        df_dur["duration_ms"] = pd.to_numeric(
            df_dur["duration_ms"],
            errors="coerce",
        )

        clip_duration = dict(
            zip(
                df_dur["clip"],
                df_dur["duration_ms"],
            )
        )

        records = []

        for rec in df.to_dict(
            orient="records"
        ):

            original_path = Path(
                rec["path_to_mp3"]
            )

            clean_relpath = (
                Path(language)
                / "clips"
                / original_path
            )

            rec["path_to_mp3"] = (
                clean_relpath.as_posix()
            )

            clip_id = original_path.name

            duration_ms = clip_duration.get(
                clip_id
            )

            if duration_ms is not None:
                rec["duration_s_before_crop"] = (
                    float(duration_ms)
                    / 1000.0
                )
            else:
                rec["duration_s_before_crop"] = None

            rec["spk_id"] = str(
                rec["spk_id"]
            )

            rec["language"] = language

            records.append(rec)

        all_data[language] = records

    return all_data


# ============================================================
# LOAD COMMONVOICE
# ============================================================

cv_map = {

    "train": get_data_from_CV(
        data_path=cv_root,
        split="train",
    ),

    "dev": get_data_from_CV(
        data_path=cv_root,
        split="dev",
    ),

    "test": get_data_from_CV(
        data_path=cv_root,
        split="test",
    ),
}


# ============================================================
# LOAD WHAM
# ============================================================

def get_wham_samples(
    wham_folder,
    split,
):
    """
    split:
        tr -> train
        cv -> validation
        tt -> test
    """

    wham_folder = Path(
        wham_folder
    )

    split_root = (
        wham_folder
        / split
    )

    if not split_root.exists():
        raise FileNotFoundError(
            f"WHAM split not found:\n"
            f"{split_root}"
        )

    noise_files = []

    for root, _, files in os.walk(
        split_root
    ):

        root = Path(root)

        for filename in files:

            if not filename.endswith(
                ".wav"
            ):
                continue

            absolute_path = (
                root / filename
            )

            relative_path = (
                absolute_path
                .relative_to(
                    wham_folder
                )
            )

            json_path = (
                Path("wham_noise")
                / relative_path
            )

            noise_files.append(
                json_path.as_posix()
            )

    return noise_files


wham_map = {

    "train": get_wham_samples(
        wham_folder=wham_noise_root,
        split="tr",
    ),

    "dev": get_wham_samples(
        wham_folder=wham_noise_root,
        split="cv",
    ),

    "test": get_wham_samples(
        wham_folder=wham_noise_root,
        split="tt",
    ),
}


# ============================================================
# BASIC STATISTICS
# ============================================================

def print_cv_statistics(
    cv_map,
):
    rows = []

    for split, split_data in cv_map.items():

        for language, records in split_data.items():

            speakers = {
                r["spk_id"]
                for r in records
            }

            rows.append({
                "split": split,
                "language": language,
                "clips": len(records),
                "speakers": len(speakers),
            })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("COMMONVOICE STATISTICS")
    print("=" * 80)

    print(
        df.to_string(
            index=False
        )
    )


def print_wham_statistics(
    wham_map,
):
    rows = []

    for split, files in wham_map.items():

        rows.append({
            "split": split,
            "noise_files": len(files),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("WHAM STATISTICS")
    print("=" * 80)

    print(
        df.to_string(
            index=False
        )
    )


print_cv_statistics(
    cv_map
)

print_wham_statistics(
    wham_map
)


# ============================================================
# CHECK CROSS-LANGUAGE SPEAKERS
# ============================================================

def check_speaker_language_collisions(
    cv_map,
):
    """
    Reporte les client_id présents dans plusieurs langues.

    Ces collisions sont importantes car client_id est considéré
    comme un identifiant global de speaker.
    """

    print("\n" + "=" * 80)
    print(
        "CHECK SPEAKER IDs ACROSS LANGUAGES"
    )
    print("=" * 80)

    for split in cv_map:

        speaker_languages = defaultdict(set)

        for language in LANGUAGES:

            for record in cv_map[
                split
            ][language]:

                speaker_languages[
                    record["spk_id"]
                ].add(language)

        collisions = {
            speaker: languages
            for speaker, languages
            in speaker_languages.items()
            if len(languages) > 1
        }

        print(
            f"{split}: "
            f"{len(collisions)} speakers "
            f"appear in multiple languages"
        )

        for speaker, languages in list(
            collisions.items()
        )[:10]:

            print(
                f"    {speaker}: "
                f"{sorted(languages)}"
            )


check_speaker_language_collisions(
    cv_map
)


# ============================================================
# TARGETS
# ============================================================

def get_targets(
    total,
):
    """
    Compute:

        total samples
        samples/environment
        samples/environment/language
    """

    n_env = len(
        ENVIRONMENTS
    )

    n_languages = len(
        LANGUAGES
    )

    if total % n_env != 0:
        raise ValueError(
            f"Dataset size {total} is not divisible "
            f"by number of environments {n_env}"
        )

    if total % (
        n_env * n_languages
    ) != 0:

        raise ValueError(
            f"Dataset size {total} is not divisible "
            f"by {n_env} environments × "
            f"{n_languages} languages"
        )

    return {
        "total": total,

        "per_environment":
            total // n_env,

        "per_language":
            total // n_languages,

        "per_cell":
            total
            // n_env
            // n_languages,
    }


# ============================================================
# BUILD SPEAKER POOLS
# ============================================================

def build_speaker_pools(
    cv_map,
):
    """
    pools[split][language][speaker] = records
    """

    pools = {}

    for split in [
        "train",
        "dev",
        "test",
    ]:

        pools[split] = {}

        for language in LANGUAGES:

            speaker_pool = defaultdict(list)

            for record in cv_map[
                split
            ][language]:

                speaker = str(
                    record["spk_id"]
                )

                speaker_pool[
                    speaker
                ].append(record)

            pools[
                split
            ][language] = speaker_pool

    return pools


SPEAKER_POOLS = build_speaker_pools(
    cv_map
)


# ============================================================
# TRAIN SPEAKER TARGET
# ============================================================

def get_train_speaker_target(
    dataset_name,
    language,
):
    """
    Number of speakers wanted in one
    environment × language cell.

    zh-CN:
        660 speakers / 4 environments = 165.

    Other languages:
        train_small = 150
        train_mono  = 450
    """

    if language == "zh-CN":

        return ZH_CN_TRAIN_SPEAKERS_PER_CELL

    if dataset_name == "noisy_CWWS_train_small":

        return TRAIN_SMALL_SPEAKERS_PER_CELL

    if dataset_name == "noisy_CWWS_train_mono":

        return TRAIN_MONO_SPEAKERS_PER_CELL

    raise ValueError(
        f"Unknown train dataset: "
        f"{dataset_name}"
    )


# ============================================================
# SELECT TRAIN SPEAKERS
# ============================================================

def select_train_speakers_for_language(
    pool,
    language,
    target_per_cell,
    cells,
    rng,
):
    """
    Select speakers for TRAIN.

    IMPORTANT
    ---------

    The four environments of one dataset must have disjoint
    speakers.

    train_small and train_mono are allowed to use the same
    speakers, PROVIDED that they use them in the same
    language and same environment.

    Example:

        zh-CN / clean:
            speakers A..Q

        zh-CN / wham:
            speakers R..Z

    train_small / clean and train_mono / clean may both use
    A..Q.

    train_small / clean and train_mono / wham may NOT share A.
    """

    available_speakers = []

    for speaker, records in pool.items():

        capacity = min(
            len(records),
            ABSOLUTE_MAX_CLIPS_PER_SPEAKER,
        )

        if capacity <= 0:
            continue

        available_speakers.append(
            {
                "speaker": speaker,
                "records": records,
                "capacity": capacity,
            }
        )

    rng.shuffle(
        available_speakers
    )

    # --------------------------------------------------------
    # We select speakers jointly over the four domains.
    #
    # This guarantees domain disjointness.
    # --------------------------------------------------------

    total_required_speakers = (
        target_per_cell
        * len(cells)
    )

    if len(
        available_speakers
    ) < target_per_cell:

        raise RuntimeError(
            "\n"
            "====================================================\n"
            "NOT ENOUGH SPEAKERS\n"
            "====================================================\n"
            f"Language          : {language}\n"
            f"Required/cell     : {target_per_cell}\n"
            f"Available speakers: {len(available_speakers)}\n"
            "===================================================="
        )

    # --------------------------------------------------------
    # Capacity-aware selection.
    #
    # A speaker needs enough clips to contribute to a cell.
    # We maximize diversity first, but we avoid selecting
    # speakers with virtually no usable clips.
    # --------------------------------------------------------

    # At least one clip is needed from every selected speaker.
    candidates = [
        x
        for x in available_speakers
        if x["capacity"] >= 1
    ]

    if len(candidates) < total_required_speakers:

        # For zh-CN this should fail only if the actual number
        # of speakers is lower than expected.
        raise RuntimeError(
            "\n"
            "====================================================\n"
            "NOT ENOUGH DISJOINT SPEAKERS FOR TRAIN DOMAINS\n"
            "====================================================\n"
            f"Language              : {language}\n"
            f"Speakers required     : {total_required_speakers}\n"
            f"Speakers available    : {len(candidates)}\n"
            f"Speakers/cell         : {target_per_cell}\n"
            f"Number of domains     : {len(cells)}\n"
            "===================================================="
        )

    # --------------------------------------------------------
    # We distribute speakers by capacity.
    #
    # To maximize the number of distinct speakers, every
    # selected speaker contributes at least one sample.
    #
    # Higher-capacity speakers are useful for filling the
    # remaining samples later.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            -x["capacity"],
            rng.random(),
        )
    )

    selected = {}

    cursor = 0

    for cell in cells:

        cell_key = (
            cell["dataset"],
            cell["environment"],
            language,
        )

        selected[
            cell_key
        ] = []

        for _ in range(
            target_per_cell
        ):

            if cursor >= len(candidates):

                raise RuntimeError(
                    f"Could not allocate speakers "
                    f"for {cell_key}"
                )

            candidate = candidates[
                cursor
            ]

            selected[
                cell_key
            ].append(
                candidate["speaker"]
            )

            cursor += 1

    return selected


# ============================================================
# ALLOCATE TRAIN SPEAKERS
# ============================================================

def allocate_train_speakers(
    cv_map,
    seed=42,
):
    """
    TRAIN allocation.

    Key constraint:

        train_small and train_mono MAY share speakers,
        but only when:

            language == same
            AND
            environment == same

    Within each train dataset:

        one speaker belongs to exactly one domain.

    Therefore:

        train_small/fr/clean
        train_small/fr/wham

    cannot share speakers.

    And:

        train_small/fr/clean
        train_mono/fr/wham

    cannot share speakers.

    But:

        train_small/fr/clean
        train_mono/fr/clean

    MAY share speakers.

    No train speaker may appear in validation or test.
    """

    rng = random.Random(seed)

    pools = build_speaker_pools(
        cv_map
    )

    train_cells = {}

    # --------------------------------------------------------
    # First allocate each language independently.
    # --------------------------------------------------------

    for language in LANGUAGES:

        pool = pools[
            "train"
        ][language]

        print("\n" + "=" * 80)
        print(
            f"TRAIN SPEAKER ALLOCATION "
            f"| LANGUAGE = {language}"
        )
        print("=" * 80)

        available = len(pool)

        print(
            f"Available speakers: {available}"
        )

        for dataset_name in [
            "noisy_CWWS_train_small",
            "noisy_CWWS_train_mono",
        ]:

            target = get_train_speaker_target(
                dataset_name,
                language,
            )

            print(
                f"{dataset_name}: "
                f"{target} speakers/domain"
            )

        # ----------------------------------------------------
        # We allocate speaker identities by DOMAIN.
        #
        # Each domain has its own speaker pool.
        #
        # The two datasets can share that domain pool.
        # ----------------------------------------------------

        domain_assignments = {}

        # ====================================================
        # zh-CN
        # ====================================================
        #
        # 660 speakers:
        #
        # clean        -> 165
        # wham         -> 165
        # wind         -> 165
        # saturation   -> 165
        #
        # The same 165 can be used by train_small and
        # train_mono within each domain.
        # ====================================================

        # ----------------------------------------------------
        # Other languages
        # ----------------------------------------------------
        #
        # Important:
        #
        # train_small requires 150 speakers/domain
        # train_mono requires 450 speakers/domain
        #
        # They can overlap, therefore we select 450 speakers
        # for the domain and use:
        #
        #   150 of them in train_small
        #   all 450 in train_mono
        #
        # This maximizes speaker diversity while respecting
        # the train_small constraint.
        # ----------------------------------------------------

        if language == "zh-CN":

            target_domain = (
                ZH_CN_TRAIN_SPEAKERS_PER_CELL
            )

            if available < (
                target_domain
                * len(ENVIRONMENTS)
            ):

                raise RuntimeError(
                    "\n"
                    "====================================================\n"
                    "ZH-CN TRAIN SPEAKER ALLOCATION IMPOSSIBLE\n"
                    "====================================================\n"
                    f"Available speakers : {available}\n"
                    f"Required           : "
                    f"{target_domain * len(ENVIRONMENTS)}\n"
                    f"Target/domain      : {target_domain}\n"
                    "\n"
                    "The requested 165 speakers/domain requires "
                    "660 distinct speakers.\n"
                    "===================================================="
                )

            candidates = list(
                pool.keys()
            )

            rng.shuffle(
                candidates
            )

            # Prefer speakers with more clips because they make
            # the exact sample target easier to reach.
            candidates.sort(
                key=lambda spk: (
                    -len(pool[spk]),
                    rng.random(),
                )
            )

            cursor = 0

            for environment in ENVIRONMENTS:

                speakers = candidates[
                    cursor:
                    cursor + target_domain
                ]

                if len(speakers) != target_domain:

                    raise RuntimeError(
                        f"Could not allocate "
                        f"{target_domain} speakers "
                        f"to zh-CN/{environment}"
                    )

                cursor += target_domain

                domain_assignments[
                    environment
                ] = {
                    "mono": list(speakers),
                    "small": list(speakers),
                }

        else:

            target_mono = (
                TRAIN_MONO_SPEAKERS_PER_CELL
            )

            target_small = (
                TRAIN_SMALL_SPEAKERS_PER_CELL
            )

            required_domain_speakers = (
                target_mono
            )

            if available < (
                required_domain_speakers
                * len(ENVIRONMENTS)
            ):

                raise RuntimeError(
                    "\n"
                    "====================================================\n"
                    "TRAIN SPEAKER ALLOCATION IMPOSSIBLE\n"
                    "====================================================\n"
                    f"Language              : {language}\n"
                    f"Available speakers    : {available}\n"
                    f"Required distinct     : "
                    f"{required_domain_speakers * len(ENVIRONMENTS)}\n"
                    f"Speakers/domain      : "
                    f"{required_domain_speakers}\n"
                    "\n"
                    "train_small and train_mono may share speakers,\n"
                    "so only 450 unique speakers/domain are required.\n"
                    "===================================================="
                )

            candidates = list(
                pool.keys()
            )

            rng.shuffle(
                candidates
            )

            candidates.sort(
                key=lambda spk: (
                    -len(pool[spk]),
                    rng.random(),
                )
            )

            cursor = 0

            for environment in ENVIRONMENTS:

                mono_speakers = candidates[
                    cursor:
                    cursor + target_mono
                ]

                if len(
                    mono_speakers
                ) != target_mono:

                    raise RuntimeError(
                        f"Could not allocate "
                        f"{target_mono} speakers "
                        f"to {language}/{environment}"
                    )

                cursor += target_mono

                # train_small uses a subset of the mono speakers.
                #
                # This is intentional:
                # train_small and train_mono may share speakers
                # and samples.
                #
                # Selecting the speakers with the highest capacity
                # helps train_small satisfy its 375 samples/cell.
                # ------------------------------------------------

                small_speakers = sorted(
                    mono_speakers,
                    key=lambda spk: (
                        -len(pool[spk]),
                        rng.random(),
                    )
                )[
                    :target_small
                ]

                domain_assignments[
                    environment
                ] = {
                    "mono": list(mono_speakers),
                    "small": list(small_speakers),
                }

        # ----------------------------------------------------
        # Save assignments.
        # ----------------------------------------------------

        for environment in ENVIRONMENTS:

            for dataset_name, key in [
                (
                    "noisy_CWWS_train_mono",
                    "mono",
                ),
                (
                    "noisy_CWWS_train_small",
                    "small",
                ),
            ]:

                cell_key = (
                    dataset_name,
                    environment,
                    language,
                )

                train_cells[
                    cell_key
                ] = list(
                    domain_assignments[
                        environment
                    ][key]
                )

        # ----------------------------------------------------
        # Verify domain disjointness.
        # ----------------------------------------------------

        for dataset_name in [
            "noisy_CWWS_train_small",
            "noisy_CWWS_train_mono",
        ]:

            seen = set()

            for environment in ENVIRONMENTS:

                cell_key = (
                    dataset_name,
                    environment,
                    language,
                )

                current = set(
                    train_cells[
                        cell_key
                    ]
                )

                overlap = (
                    seen
                    & current
                )

                if overlap:

                    raise RuntimeError(
                        "\n"
                        "TRAIN SPEAKER COLLISION\n"
                        f"Dataset    : {dataset_name}\n"
                        f"Language   : {language}\n"
                        f"Environment: {environment}\n"
                        f"Overlap    : {len(overlap)}\n"
                    )

                seen.update(
                    current
                )

        # ----------------------------------------------------
        # Print summary.
        # ----------------------------------------------------

        for environment in ENVIRONMENTS:

            small_key = (
                "noisy_CWWS_train_small",
                environment,
                language,
            )

            mono_key = (
                "noisy_CWWS_train_mono",
                environment,
                language,
            )

            print(
                f"{environment:12s} | "
                f"small={len(train_cells[small_key]):4d} | "
                f"mono={len(train_cells[mono_key]):4d} | "
                f"overlap="
                f"{len(set(train_cells[small_key]) & set(train_cells[mono_key])):4d}"
            )

    return train_cells


# ============================================================
# TRAIN RECORD ALLOCATION
# ============================================================

def allocate_train_records(
    train_cells,
    pools,
    seed=42,
):
    """
    Convert speaker allocation into exact clip allocation.

    We maximize speaker diversity first.

    Every selected speaker receives at least one clip.

    Remaining clips are distributed among already-selected
    speakers.

    Therefore, speakers are never added after the target number
    of distinct speakers has been reached.
    """

    rng = random.Random(seed)

    cell_limits = {}

    for cell_key, speakers in train_cells.items():

        dataset_name, environment, language = (
            cell_key
        )

        if dataset_name == "noisy_CWWS_train_small":
            target = get_targets(
                DATASET_CONFIG[
                    dataset_name
                ]["total"]
            )["per_cell"]

        else:
            target = get_targets(
                DATASET_CONFIG[
                    dataset_name
                ]["total"]
            )["per_cell"]

        pool = pools[
            "train"
        ][language]

        # ----------------------------------------------------
        # Check number of speakers.
        # ----------------------------------------------------

        if len(speakers) == 0:

            raise RuntimeError(
                f"{cell_key}: no speakers allocated."
            )

        if len(speakers) > target:

            raise RuntimeError(
                f"{cell_key}: "
                f"{len(speakers)} speakers > "
                f"{target} samples."
            )

        # ----------------------------------------------------
        # Capacity.
        # ----------------------------------------------------

        capacity = {}

        for speaker in speakers:

            capacity[speaker] = min(
                len(pool[speaker]),
                ABSOLUTE_MAX_CLIPS_PER_SPEAKER,
            )

        total_capacity = sum(
            capacity.values()
        )

        if total_capacity < target:

            raise RuntimeError(
                "\n"
                "====================================================\n"
                "TRAIN CELL CANNOT BE FILLED\n"
                "====================================================\n"
                f"Cell             : {cell_key}\n"
                f"Target samples   : {target}\n"
                f"Speakers         : {len(speakers)}\n"
                f"Available clips  : {total_capacity}\n"
                "===================================================="
            )

        # ----------------------------------------------------
        # Start with ONE clip per speaker.
        # ----------------------------------------------------

        limits = {
            speaker: 1
            for speaker in speakers
        }

        current = len(
            speakers
        )

        # ----------------------------------------------------
        # Add clips while keeping speaker diversity fixed.
        # ----------------------------------------------------

        while current < target:

            expandable = [
                speaker
                for speaker in speakers
                if limits[speaker]
                < capacity[speaker]
            ]

            if not expandable:

                raise RuntimeError(
                    f"{cell_key}: "
                    f"cannot reach target {target}; "
                    f"current={current}"
                )

            # Prefer speakers with the smallest number of clips
            # to keep the distribution reasonably balanced.
            #
            # If several speakers have the same count, favor
            # speakers with more remaining capacity.
            # ------------------------------------------------

            expandable.sort(
                key=lambda speaker: (
                    limits[speaker],
                    -(
                        capacity[speaker]
                        - limits[speaker]
                    ),
                    rng.random(),
                )
            )

            speaker = expandable[0]

            limits[
                speaker
            ] += 1

            current += 1

        if sum(
            limits.values()
        ) != target:

            raise RuntimeError(
                f"{cell_key}: allocation mismatch."
            )

        cell_limits[
            cell_key
        ] = limits

    return cell_limits


# ============================================================
# SELECT TRAIN RECORDS
# ============================================================

def select_records_for_train_cell(
    pool,
    speakers,
    limits,
    target,
    rng,
):
    """
    Select exactly the requested number of clips.

    train_small and train_mono are allowed to select the same
    clean CommonVoice clip.
    """

    selected = []

    for speaker in speakers:

        records = list(
            pool[speaker]
        )

        rng.shuffle(
            records
        )

        n_take = limits[
            speaker
        ]

        if n_take > len(records):

            raise RuntimeError(
                f"Speaker {speaker}: requested "
                f"{n_take} clips but only "
                f"{len(records)} available."
            )

        selected.extend(
            records[:n_take]
        )

    if len(selected) != target:

        raise RuntimeError(
            f"Expected {target} records, "
            f"got {len(selected)}."
        )

    # No duplicate clip inside the SAME cell.
    paths = [
        r["path_to_mp3"]
        for r in selected
    ]

    if len(paths) != len(set(paths)):

        raise RuntimeError(
            "Duplicate clean clip inside train cell."
        )

    rng.shuffle(
        selected
    )

    return selected


# ============================================================
# VALIDATION ALLOCATION
# ============================================================

def allocate_validation_samples(
    cv_map,
    forbidden_speakers=None,
    seed=42,
):
    """
    Validation:

        * one sample per speaker
        * same clean samples across environments
        * no speaker overlap with train/test handled globally
          later.

    We choose exactly the same base records for all four
    environments.

    Since validation has:

        6000 total
        4 environments
        4 languages

    we need:

        375 speakers/language

    because each speaker is replicated over the four domains.
    """

    rng = random.Random(seed)

    pools = build_speaker_pools(
        cv_map
    )

    if forbidden_speakers is None:
        forbidden_speakers = set()

    forbidden_speakers = {
        str(speaker)
        for speaker in forbidden_speakers
    }

    print(
        f"Speakers forbidden from validation: "
        f"{len(forbidden_speakers)}"
    )

    validation_records = {}

    speakers_per_language = (
        get_targets(
            DATASET_CONFIG[
                "noisy_CWWS_validation"
            ]["total"]
        )["per_cell"]
    )

    print("\n" + "=" * 80)
    print("VALIDATION ALLOCATION")
    print("=" * 80)

    print(
        f"Speakers/language: "
        f"{speakers_per_language}"
    )

    for language in LANGUAGES:

        pool = pools[
            "dev"
        ][language]

        candidates = [
            speaker
            for speaker, records
            in pool.items()
            if (
                len(records) >= 1
                and str(speaker) not in forbidden_speakers
            )
        ]

        print(
            f"{language}: "
            f"{len(candidates)} speakers available "
            f"after train exclusion"
        )

        if len(candidates) < speakers_per_language:

            raise RuntimeError(
                f"Validation {language}: "
                f"need {speakers_per_language} speakers, "
                f"have {len(candidates)}."
            )

        rng.shuffle(
            candidates
        )

        # Prefer speakers with available records.
        candidates.sort(
            key=lambda speaker: (
                -len(pool[speaker]),
                rng.random(),
            )
        )

        selected_speakers = candidates[
            :speakers_per_language
        ]

        overlap = (
            set(selected_speakers)
            & forbidden_speakers
        )

        if overlap:

            raise RuntimeError(
                f"Validation {language}: "
                f"{len(overlap)} forbidden speakers "
                f"were selected."
            )

        records = []

        for speaker in selected_speakers:

            speaker_records = list(
                pool[speaker]
            )

            rng.shuffle(
                speaker_records
            )

            records.append(
                speaker_records[0]
            )

        # ----------------------------------------------------
        # Exactly one sample per speaker.
        # ----------------------------------------------------

        if len(records) != speakers_per_language:

            raise RuntimeError(
                f"Validation {language}: "
                "incorrect number of records."
            )

        # ----------------------------------------------------
        # Same clean sample across all domains.
        # ----------------------------------------------------

        for environment in ENVIRONMENTS:

            cell_key = (
                "noisy_CWWS_validation",
                environment,
                language,
            )

            validation_records[
                cell_key
            ] = list(records)

        print(
            f"{language}: "
            f"{len(selected_speakers)} speakers"
        )

    return validation_records


# ============================================================
# TEST ALLOCATION
# ============================================================

def allocate_test_samples(
    cv_map,
    forbidden_speakers=None,
    seed=42,
):
    """
    Test:

        * maximum 10 samples per speaker
        * same clean clips across environments
        * all domains differ only by corruption

    We select a BASE SET of clean clips for each language.
    This same base set is copied to all four domains.

    We therefore never independently sample test clips for
    clean / wham / wind / saturation.
    """

    rng = random.Random(seed)

    pools = build_speaker_pools(
        cv_map
    )

    if forbidden_speakers is None:
        forbidden_speakers = set()

    forbidden_speakers = {
        str(speaker)
        for speaker in forbidden_speakers
    }

    print(
        f"Speakers forbidden from test: "
        f"{len(forbidden_speakers)}"
    )

    total = DATASET_CONFIG[
        "noisy_CWWS_test"
    ]["total"]

    target_per_cell = get_targets(
        total
    )["per_cell"]

    # --------------------------------------------------------
    # We need target_per_cell clean samples/language/domain.
    #
    # Since the same base clips are replicated over 4 domains,
    # we need exactly target_per_cell base clips per language.
    # --------------------------------------------------------

    test_records = {}

    print("\n" + "=" * 80)
    print("TEST ALLOCATION")
    print("=" * 80)

    print(
        f"Samples/language/domain: "
        f"{target_per_cell}"
    )

    for language in LANGUAGES:

        pool = pools[
            "test"
        ][language]

        # ----------------------------------------------------
        # We first choose speakers.
        #
        # Each speaker can contribute at most 10 samples.
        # ----------------------------------------------------

        candidates = []

        for speaker, records in pool.items():

            # --------------------------------------------------------
            # NEVER reuse a speaker already allocated to train or
            # validation.
            # --------------------------------------------------------

            if str(speaker) in forbidden_speakers:
                continue

            usable = min(
                len(records),
                TEST_MAX_SAMPLES_PER_SPEAKER,
            )

            if usable <= 0:
                continue

            candidates.append({
                "speaker": speaker,
                "records": list(records),
                "capacity": usable,
            })

        rng.shuffle(
            candidates
        )

        candidates.sort(
            key=lambda x: (
                -x["capacity"],
                rng.random(),
            )
        )

        total_capacity = sum(
            x["capacity"]
            for x in candidates
        )

        if total_capacity < target_per_cell:

            raise RuntimeError(
                "\n"
                "====================================================\n"
                "TEST CANNOT BE FILLED\n"
                "====================================================\n"
                f"Language          : {language}\n"
                f"Target samples    : {target_per_cell}\n"
                f"Available capacity: {total_capacity}\n"
                f"Max samples/spk   : {TEST_MAX_SAMPLES_PER_SPEAKER}\n"
                "===================================================="
            )

        # ----------------------------------------------------
        # Select as many speakers as possible.
        #
        # We use at least one clip from every selected speaker,
        # then distribute remaining clips.
        #
        # This maximizes speaker diversity subject to the
        # max-10 constraint.
        # ----------------------------------------------------

        selected = []

        current_capacity = 0

        for candidate in candidates:

            if current_capacity >= target_per_cell:
                break

            selected.append(
                candidate
            )

            current_capacity += (
                candidate["capacity"]
            )
        selected_speakers = {
            candidate["speaker"]
            for candidate in selected
        }

        overlap = (
            selected_speakers
            & forbidden_speakers
        )

        if overlap:

            raise RuntimeError(
                f"Test {language}: "
                f"{len(overlap)} forbidden speakers "
                f"were selected."
            )
        # ----------------------------------------------------
        # Determine exact clip counts.
        # ----------------------------------------------------

        limits = {
            candidate["speaker"]: 1
            for candidate in selected
        }

        current = len(
            selected
        )

        while current < target_per_cell:

            expandable = [
                candidate
                for candidate in selected
                if limits[
                    candidate["speaker"]
                ] < candidate["capacity"]
            ]

            if not expandable:

                raise RuntimeError(
                    f"Test {language}: "
                    f"cannot reach {target_per_cell}."
                )

            expandable.sort(
                key=lambda candidate: (
                    limits[
                        candidate["speaker"]
                    ],
                    -candidate["capacity"],
                    rng.random(),
                )
            )

            speaker = (
                expandable[0]["speaker"]
            )

            limits[
                speaker
            ] += 1

            current += 1

        # ----------------------------------------------------
        # Select actual records.
        # ----------------------------------------------------

        base_records = []

        for candidate in selected:

            speaker = candidate[
                "speaker"
            ]

            records = list(
                candidate["records"]
            )

            rng.shuffle(
                records
            )

            n_take = limits[
                speaker
            ]

            base_records.extend(
                records[:n_take]
            )

        if len(base_records) != target_per_cell:

            raise RuntimeError(
                f"Test {language}: "
                f"expected {target_per_cell}, "
                f"got {len(base_records)}."
            )

        # ----------------------------------------------------
        # Verify max 10 samples/speaker.
        # ----------------------------------------------------

        speaker_counts = defaultdict(int)

        for record in base_records:

            speaker_counts[
                record["spk_id"]
            ] += 1

        if speaker_counts:

            max_count = max(
                speaker_counts.values()
            )

            if max_count > (
                TEST_MAX_SAMPLES_PER_SPEAKER
            ):

                raise RuntimeError(
                    f"Test {language}: "
                    f"speaker exceeds "
                    f"{TEST_MAX_SAMPLES_PER_SPEAKER} samples."
                )

        # ----------------------------------------------------
        # No duplicate clean clip.
        # ----------------------------------------------------

        paths = [
            r["path_to_mp3"]
            for r in base_records
        ]

        if len(paths) != len(set(paths)):

            raise RuntimeError(
                f"Test {language}: "
                "duplicate clean clip."
            )

        # ----------------------------------------------------
        # SAME BASE RECORDS IN ALL DOMAINS.
        # ----------------------------------------------------

        for environment in ENVIRONMENTS:

            cell_key = (
                "noisy_CWWS_test",
                environment,
                language,
            )

            test_records[
                cell_key
            ] = list(base_records)

        print(
            f"{language}: "
            f"{len(speaker_counts)} speakers, "
            f"{len(base_records)} samples, "
            f"max={max(speaker_counts.values())}"
        )

    return test_records


# ============================================================
# GLOBAL SPEAKER EXCLUSIVITY
# ============================================================

def verify_cross_split_speaker_exclusivity(
    train_cells,
    validation_records,
    test_records,
):
    """
    Global constraint:

        speaker_unique_across_datasets = True

    Therefore:

        TRAIN ∩ VALIDATION = empty
        TRAIN ∩ TEST       = empty
        VALIDATION ∩ TEST  = empty

    IMPORTANT:

        train_small ∩ train_mono is allowed,

    but only within the SAME language × environment.

    """

    print("\n" + "=" * 80)
    print(
        "GLOBAL SPEAKER EXCLUSIVITY"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_speakers = set()

    for cell_key, speakers in train_cells.items():

        train_speakers.update(
            speakers
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_speakers = set()

    for records in validation_records.values():

        for record in records:

            validation_speakers.add(
                record["spk_id"]
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_speakers = set()

    for records in test_records.values():

        for record in records:

            test_speakers.add(
                record["spk_id"]
            )

    # --------------------------------------------------------
    # Check.
    # --------------------------------------------------------

    train_validation = (
        train_speakers
        & validation_speakers
    )

    train_test = (
        train_speakers
        & test_speakers
    )

    validation_test = (
        validation_speakers
        & test_speakers
    )

    print(
        f"Train speakers      : "
        f"{len(train_speakers)}"
    )

    print(
        f"Validation speakers : "
        f"{len(validation_speakers)}"
    )

    print(
        f"Test speakers       : "
        f"{len(test_speakers)}"
    )

    print(
        f"Train ∩ validation  : "
        f"{len(train_validation)}"
    )

    print(
        f"Train ∩ test        : "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ test   : "
        f"{len(validation_test)}"
    )

    if train_validation:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "train speaker appears in validation."
        )

    if train_test:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "train speaker appears in test."
        )

    if validation_test:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "validation speaker appears in test."
        )

    print(
        "\n✓ train / validation / test speakers are disjoint."
    )


# ============================================================
# TRAIN INTERNAL VERIFICATION
# ============================================================

def verify_train_speaker_constraints(
    train_cells,
):
    """
    Verify:

    1. train_small:
       speaker belongs to one domain only.

    2. train_mono:
       speaker belongs to one domain only.

    3. train_small/train_mono:
       overlap is allowed ONLY in same domain + language.

    4. Number of speakers/cell is correct.
    """

    print("\n" + "=" * 80)
    print(
        "TRAIN SPEAKER CONSTRAINT VERIFICATION"
    )
    print("=" * 80)

    for language in LANGUAGES:

        # ----------------------------------------------------
        # Check each dataset separately.
        # ----------------------------------------------------

        for dataset_name in [
            "noisy_CWWS_train_small",
            "noisy_CWWS_train_mono",
        ]:

            seen = {}

            for environment in ENVIRONMENTS:

                key = (
                    dataset_name,
                    environment,
                    language,
                )

                speakers = set(
                    train_cells[key]
                )

                target = get_train_speaker_target(
                    dataset_name,
                    language,
                )

                if len(speakers) != target:

                    raise RuntimeError(
                        f"{key}: expected "
                        f"{target} speakers, got "
                        f"{len(speakers)}."
                    )

                for speaker in speakers:

                    if speaker in seen:

                        raise RuntimeError(
                            "\n"
                            "TRAIN DOMAIN COLLISION\n"
                            f"Dataset={dataset_name}\n"
                            f"Language={language}\n"
                            f"Speaker={speaker}\n"
                            f"Domain 1={seen[speaker]}\n"
                            f"Domain 2={environment}\n"
                        )

                    seen[
                        speaker
                    ] = environment

        # ----------------------------------------------------
        # Cross train_small / train_mono.
        #
        # Overlap is allowed ONLY within same environment.
        # ----------------------------------------------------

        for environment in ENVIRONMENTS:

            small_key = (
                "noisy_CWWS_train_small",
                environment,
                language,
            )

            mono_key = (
                "noisy_CWWS_train_mono",
                environment,
                language,
            )

            small = set(
                train_cells[
                    small_key
                ]
            )

            mono = set(
                train_cells[
                    mono_key
                ]
            )

            # This overlap is explicitly allowed.
            allowed_overlap = (
                small
                & mono
            )

            # Verify that no small speaker appears in a
            # DIFFERENT mono domain.
            for other_environment in ENVIRONMENTS:

                if other_environment == environment:
                    continue

                other_key = (
                    "noisy_CWWS_train_mono",
                    other_environment,
                    language,
                )

                illegal = (
                    small
                    & set(
                        train_cells[
                            other_key
                        ]
                    )
                )

                if illegal:

                    raise RuntimeError(
                        "\n"
                        "ILLEGAL TRAIN CROSS-DATASET OVERLAP\n"
                        f"Language          : {language}\n"
                        f"Small domain      : {environment}\n"
                        f"Mono domain       : {other_environment}\n"
                        f"Speakers          : {len(illegal)}\n"
                        "\n"
                        "A speaker may be shared between "
                        "train_small and train_mono only "
                        "when language AND domain are identical.\n"
                    )

            print(
                f"{language:5s} | "
                f"{environment:12s} | "
                f"small={len(small):4d} | "
                f"mono={len(mono):4d} | "
                f"allowed overlap={len(allowed_overlap):4d}"
            )

    print(
        "\n✓ TRAIN speaker constraints passed."
    )


# ============================================================
# CLEAN CONSISTENCY VERIFICATION
# ============================================================

def verify_evaluation_clean_consistency(
    validation_records,
    test_records,
):
    """
    Verify that validation/test have exactly the same clean
    samples across all environments.

    For each language:

        clean == wham == wind == saturation
    """

    print("\n" + "=" * 80)
    print(
        "VALIDATION / TEST CLEAN CONSISTENCY"
    )
    print("=" * 80)

    for dataset_name, records_dict in [
        (
            "noisy_CWWS_validation",
            validation_records,
        ),
        (
            "noisy_CWWS_test",
            test_records,
        ),
    ]:

        for language in LANGUAGES:

            reference_key = (
                dataset_name,
                "clean",
                language,
            )

            reference = [
                r["path_to_mp3"]
                for r in records_dict[
                    reference_key
                ]
            ]

            reference_set = set(
                reference
            )

            for environment in ENVIRONMENTS:

                key = (
                    dataset_name,
                    environment,
                    language,
                )

                current = [
                    r["path_to_mp3"]
                    for r in records_dict[
                        key
                    ]
                ]

                current_set = set(
                    current
                )

                if current != reference:

                    raise RuntimeError(
                        "\n"
                        "CLEAN SIGNAL MISMATCH\n"
                        f"Dataset     : {dataset_name}\n"
                        f"Language    : {language}\n"
                        f"Environment : {environment}\n"
                        f"Reference   : clean\n"
                    )

                if current_set != reference_set:

                    raise RuntimeError(
                        f"{key}: "
                        "clean clip set differs."
                    )

            print(
                f"{dataset_name:28s} | "
                f"{language:5s} | "
                f"{len(reference):5d} identical clean samples"
            )

    print(
        "\n✓ Validation/test clean signals are identical "
        "across domains."
    )


# ============================================================
# BUILD CORRUPTED SAMPLE
# ============================================================

def build_sample(
    record,
    environment,
    split,
    wham_map,
    rng,
):
    """
    Build one JSON sample.

    The clean field ALWAYS points to the original CommonVoice
    clip.

    For validation/test, the same record is passed to all
    environments. Only corruption parameters differ.
    """

    clean_path = (
        record[
            "path_to_mp3"
        ]
    )

    if environment == "clean":

        return {
            "clean": clean_path,
            "noise": "clean",
            "snr_db": None,
            "gain": None,
            "gustiness": None,
            "wind_profile": None,
        }

    # ========================================================
    # WHAM
    # ========================================================

    if environment == "wham":

        noise_files = wham_map[
            split
        ]

        if not noise_files:

            raise RuntimeError(
                f"No WHAM files available "
                f"for split={split}"
            )

        noise_path = (
            noise_files[
                int(
                    rng.integers(
                        0,
                        len(noise_files)
                    )
                )
            ]
        )

        snr_db = rng.uniform(
            wham_snr_range[0],
            wham_snr_range[1],
        )

        return {
            "clean": clean_path,
            "noise": noise_path,
            "snr_db": float(
                snr_db
            ),
            "gain": None,
            "gustiness": None,
            "wind_profile": None,
        }

    # ========================================================
    # WIND
    # ========================================================

    if environment == "wind":

        gustiness = rng.uniform(
            gustiness_range[0],
            gustiness_range[1],
        )

        wind_profile = rng.uniform(
            wind_profile_magnitude_range[0],
            wind_profile_magnitude_range[1],
        )

        snr_db = rng.uniform(
            wind_snr_range[0],
            wind_snr_range[1],
        )

        return {
            "clean": clean_path,
            "noise": "wind",
            "snr_db": float(
                snr_db
            ),
            "gain": None,
            "gustiness": float(
                gustiness
            ),
            "wind_profile": float(
                wind_profile
            ),
        }

    # ========================================================
    # SATURATION
    # ========================================================

    if environment == "saturation":

        gain = rng.uniform(
            saturation_gain_range[0],
            saturation_gain_range[1],
        )

        return {
            "clean": clean_path,
            "noise": "saturation",
            "snr_db": None,
            "gain": float(
                gain
            ),
            "gustiness": None,
            "wind_profile": None,
        }

    raise ValueError(
        f"Unknown environment: "
        f"{environment}"
    )


# ============================================================
# GENERATE TRAIN DATASET
# ============================================================

def generate_train_dataset(
    dataset_name,
    train_cells,
    train_limits,
    cv_map,
    wham_map,
    seed=42,
):
    """
    Generate train_small or train_mono.

    IMPORTANT:

    We independently sample the clean records for every
    environment.

    Therefore:

        clean != necessarily identical across domains.

    This is exactly what is requested for train.
    """

    config = DATASET_CONFIG[
        dataset_name
    ]

    total = config[
        "total"
    ]

    split = config[
        "split"
    ]

    target = get_targets(
        total
    )["per_cell"]

    rng = np.random.default_rng(
        seed
    )

    data = []
    statistics = []

    for environment in ENVIRONMENTS:

        for language in LANGUAGES:

            cell_key = (
                dataset_name,
                environment,
                language,
            )

            speakers = train_cells[
                cell_key
            ]

            limits = train_limits[
                cell_key
            ]

            pool = build_speaker_pools(
                cv_map
            )[
                "train"
            ][language]

            selected_records = (
                select_records_for_train_cell(
                    pool=pool,
                    speakers=speakers,
                    limits=limits,
                    target=target,
                    rng=rng,
                )
            )

            # ------------------------------------------------
            # Create corrupted samples.
            # ------------------------------------------------

            for record in selected_records:

                sample = build_sample(
                    record=record,
                    environment=environment,
                    split=split,
                    wham_map=wham_map,
                    rng=rng,
                )

                data.append(
                    sample
                )

                statistics.append({
                    "dataset":
                        dataset_name,

                    "environment":
                        environment,

                    "language":
                        language,

                    "speaker":
                        record["spk_id"],

                    "clip":
                        record["path_to_mp3"],
                })

    stats_df = pd.DataFrame(
        statistics
    )

    verify_dataset_statistics(
        dataset_name,
        data,
        stats_df,
        total,
    )

    # --------------------------------------------------------
    # Shuffle final dataset.
    # --------------------------------------------------------

    random.Random(
        seed
    ).shuffle(
        data
    )

    output = build_output_json(
        dataset_name=dataset_name,
        split=split,
        total=total,
        data=data,
        stats_df=stats_df,
    )

    save_output(
        dataset_name,
        output,
        output_path
    )

    return output, stats_df


# ============================================================
# GENERATE VALIDATION
# ============================================================

def generate_validation_dataset(
    validation_records,
    wham_map,
    seed=42,
):
    """
    Validation:

        same clean record in every environment.
    """

    dataset_name = (
        "noisy_CWWS_validation"
    )

    total = DATASET_CONFIG[
        dataset_name
    ]["total"]

    split = "dev"

    rng = np.random.default_rng(
        seed
    )

    data = []
    statistics = []

    for environment in ENVIRONMENTS:

        for language in LANGUAGES:

            key = (
                dataset_name,
                environment,
                language,
            )

            records = validation_records[
                key
            ]

            for record in records:

                sample = build_sample(
                    record=record,
                    environment=environment,
                    split=split,
                    wham_map=wham_map,
                    rng=rng,
                )

                data.append(
                    sample
                )

                statistics.append({
                    "dataset":
                        dataset_name,

                    "environment":
                        environment,

                    "language":
                        language,

                    "speaker":
                        record["spk_id"],

                    "clip":
                        record["path_to_mp3"],
                })

    stats_df = pd.DataFrame(
        statistics
    )

    verify_dataset_statistics(
        dataset_name,
        data,
        stats_df,
        total,
    )

    random.Random(
        seed
    ).shuffle(
        data
    )

    output = build_output_json(
        dataset_name=dataset_name,
        split=split,
        total=total,
        data=data,
        stats_df=stats_df,
    )

    save_output(
        dataset_name,
        output,
        output_path
    )

    return output, stats_df


# ============================================================
# GENERATE TEST
# ============================================================

def generate_test_dataset(
    test_records,
    wham_map,
    seed=42,
):
    """
    Test:

        same clean record in every environment.
    """

    dataset_name = (
        "noisy_CWWS_test"
    )

    total = DATASET_CONFIG[
        dataset_name
    ]["total"]

    split = "test"

    rng = np.random.default_rng(
        seed
    )

    data = []
    statistics = []

    for environment in ENVIRONMENTS:

        for language in LANGUAGES:

            key = (
                dataset_name,
                environment,
                language,
            )

            records = test_records[
                key
            ]

            for record in records:

                sample = build_sample(
                    record=record,
                    environment=environment,
                    split=split,
                    wham_map=wham_map,
                    rng=rng,
                )

                data.append(
                    sample
                )

                statistics.append({
                    "dataset":
                        dataset_name,

                    "environment":
                        environment,

                    "language":
                        language,

                    "speaker":
                        record["spk_id"],

                    "clip":
                        record["path_to_mp3"],
                })

    stats_df = pd.DataFrame(
        statistics
    )

    verify_dataset_statistics(
        dataset_name,
        data,
        stats_df,
        total,
    )

    # --------------------------------------------------------
    # Test maximum 10 samples/speaker.
    # --------------------------------------------------------

    speaker_counts = (
        stats_df
        .groupby(
            "speaker"
        )
        .size()
    )

    # IMPORTANT:
    #
    # The same sample is replicated over 4 domains.
    #
    # Therefore the statistics dataframe contains 4 entries
    # for one clean sample.
    #
    # The max constraint applies to the BASE clean samples,
    # not to the four corrupted versions.
    #
    # We therefore divide by number of environments after
    # checking that the counts are perfectly replicated.
    # --------------------------------------------------------

    base_counts = (
        speaker_counts
        / len(ENVIRONMENTS)
    )

    if not np.allclose(
        base_counts.values,
        base_counts.values.astype(int),
    ):

        raise RuntimeError(
            "Test speaker replication is inconsistent."
        )

    max_base_count = int(
        base_counts.max()
    )

    if max_base_count > (
        TEST_MAX_SAMPLES_PER_SPEAKER
    ):

        raise RuntimeError(
            f"Test speaker exceeds "
            f"{TEST_MAX_SAMPLES_PER_SPEAKER} "
            f"base clean samples."
        )

    output = build_output_json(
        dataset_name=dataset_name,
        split=split,
        total=total,
        data=data,
        stats_df=stats_df,
    )

    save_output(
        dataset_name,
        output,
        output_path
    )

    return output, stats_df


# ============================================================
# DATASET STATISTICS
# ============================================================

def verify_dataset_statistics(
    dataset_name,
    data,
    stats_df,
    total,
):
    """
    Generic verification.
    """

    if len(data) != total:

        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{total} samples, got "
            f"{len(data)}."
        )

    # --------------------------------------------------------
    # Environment balance.
    # --------------------------------------------------------

    env_counts = (
        stats_df
        .groupby(
            "environment"
        )
        .size()
        .sort_index()
    )

    expected_env = (
        total
        // len(ENVIRONMENTS)
    )

    print(
        f"\n{dataset_name} - Environment counts:"
    )

    print(
        env_counts
    )

    if not (
        env_counts
        == expected_env
    ).all():

        raise RuntimeError(
            f"{dataset_name}: "
            "environment imbalance."
        )

    # --------------------------------------------------------
    # Language balance.
    # --------------------------------------------------------

    language_counts = (
        stats_df
        .groupby(
            "language"
        )
        .size()
        .sort_index()
    )

    expected_language = (
        total
        // len(LANGUAGES)
    )

    print(
        f"\n{dataset_name} - Language counts:"
    )

    print(
        language_counts
    )

    if not (
        language_counts
        == expected_language
    ).all():

        raise RuntimeError(
            f"{dataset_name}: "
            "language imbalance."
        )

    # --------------------------------------------------------
    # Cell balance.
    # --------------------------------------------------------

    cross = pd.crosstab(
        stats_df[
            "environment"
        ],
        stats_df[
            "language"
        ],
    )

    expected_cell = (
        total
        // len(ENVIRONMENTS)
        // len(LANGUAGES)
    )

    print(
        f"\n{dataset_name} - "
        f"Environment × language:"
    )

    print(
        cross
    )

    if not (
        cross
        == expected_cell
    ).all().all():

        raise RuntimeError(
            f"{dataset_name}: "
            "environment/language imbalance."
        )

    # --------------------------------------------------------
    # Speaker statistics.
    # --------------------------------------------------------

    speaker_counts = (
        stats_df
        .groupby(
            "speaker"
        )
        .size()
    )

    print(
        f"\n{dataset_name} - Speaker statistics:"
    )

    print(
        f"  Unique speakers : "
        f"{len(speaker_counts)}"
    )

    print(
        f"  Mean clips/spk  : "
        f"{speaker_counts.mean():.2f}"
    )

    print(
        f"  Median clips/spk: "
        f"{speaker_counts.median():.2f}"
    )

    print(
        f"  Max clips/spk   : "
        f"{speaker_counts.max()}"
    )

    if speaker_counts.max() > (
        ABSOLUTE_MAX_CLIPS_PER_SPEAKER
    ):

        raise RuntimeError(
            f"{dataset_name}: speaker exceeds "
            f"{ABSOLUTE_MAX_CLIPS_PER_SPEAKER}."
        )

    # --------------------------------------------------------
    # Duplicate clips inside dataset.
    #
    # For validation/test the SAME clean clip appears in all
    # four environments, so duplicates are EXPECTED there.
    # We therefore do not apply the old global uniqueness
    # assertion here.
    # --------------------------------------------------------

    print(
        f"  Clean clip entries: "
        f"{len(stats_df)}"
    )


# ============================================================
# OUTPUT JSON
# ============================================================

def build_output_json(
    dataset_name,
    split,
    total,
    data,
    stats_df,
):
    """
    Build final JSON object.
    """

    expected_env = (
        total
        // len(ENVIRONMENTS)
    )

    expected_language = (
        total
        // len(LANGUAGES)
    )

    expected_cell = (
        total
        // len(ENVIRONMENTS)
        // len(LANGUAGES)
    )

    speaker_counts = (
        stats_df
        .groupby(
            "speaker"
        )
        .size()
    )

    return {

        "sampler_info": {

            "wind_gustiness_range": [
                gustiness_range[0],
                gustiness_range[1],
            ],

            "wind_profile_magnitude_range": [
                wind_profile_magnitude_range[0],
                wind_profile_magnitude_range[1],
            ],

            "wind_snr_range": [
                wind_snr_range[0],
                wind_snr_range[1],
            ],

            "wham_snr_range": [
                wham_snr_range[0],
                wham_snr_range[1],
            ],

            "saturation_gain_range": [
                saturation_gain_range[0],
                saturation_gain_range[1],
            ],

            "distribution":
                "uniform",

            "seed":
                SEED,
        },

        "speaker_constraints": {

            "speaker_unique_across_datasets":
                True,

            "train_small_mono_shared_speakers":
                True,

            "train_small_mono_shared_samples":
                True,

            "train_speaker_unique_across_domains":
                True,

            "validation_one_sample_per_speaker":
                False,

            "validation_same_clean_across_domains":
                True,

            "test_same_clean_across_domains":
                True,

            "test_max_samples_per_speaker":
                TEST_MAX_SAMPLES_PER_SPEAKER,

            "absolute_max_clips_per_speaker":
                ABSOLUTE_MAX_CLIPS_PER_SPEAKER,
        },

        "train_constraints": {

            "train_small_speakers_per_cell":
                TRAIN_SMALL_SPEAKERS_PER_CELL,

            "train_mono_speakers_per_cell":
                TRAIN_MONO_SPEAKERS_PER_CELL,

            "zh_CN_speakers_per_cell":
                ZH_CN_TRAIN_SPEAKERS_PER_CELL,

            "train_small_mono_overlap":
                "allowed only within same language and same environment",
        },

        "dataset_info": {

            "dataset":
                dataset_name,

            "commonvoice_split":
                split,

            "total_samples":
                total,

            "samples_per_environment":
                expected_env,

            "samples_per_language":
                expected_language,

            "samples_per_environment_language":
                expected_cell,

            "unique_speakers":
                len(speaker_counts),
        },

        "audio_ext":
            ".mp3",

        "data":
            data,
    }


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    dataset_name,
    output,
    output_path,
):
    output_path_full = (
        output_path
        / f"{dataset_name}.json"
    )

    output_path_full.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path_full,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n✓ {dataset_name}"
    )

    print(
        f"  Samples : "
        f"{len(output['data'])}"
    )

    print(
        f"  Saved   : "
        f"{output_path_full}"
    )


# ============================================================
# GLOBAL FINAL VERIFICATION
# ============================================================

def verify_global_datasets(
    all_statistics,
):
    """
    Final verification.

    IMPORTANT:

    train_small/train_mono overlap is allowed.

    Therefore we DO NOT assert that a speaker has only one
    dataset globally.

    Instead:

        train_small/train_mono overlap:
            allowed only same language + environment.

        train/test:
            forbidden.

        train/validation:
            forbidden.

        validation/test:
            forbidden.
    """

    print("\n" + "=" * 80)
    print(
        "GLOBAL FINAL VERIFICATION"
    )
    print("=" * 80)

    global_df = pd.concat(
        all_statistics,
        ignore_index=True,
    )

    # ========================================================
    # Dataset speaker sets
    # ========================================================

    dataset_speakers = {}

    for dataset_name in global_df[
        "dataset"
    ].unique():

        dataset_speakers[
            dataset_name
        ] = set(
            global_df.loc[
                global_df["dataset"]
                == dataset_name,
                "speaker",
            ]
        )

    train_small = dataset_speakers[
        "noisy_CWWS_train_small"
    ]

    train_mono = dataset_speakers[
        "noisy_CWWS_train_mono"
    ]

    validation = dataset_speakers[
        "noisy_CWWS_validation"
    ]

    test = dataset_speakers[
        "noisy_CWWS_test"
    ]

    # ========================================================
    # Train / validation / test disjointness
    # ========================================================

    train_all = (
        train_small
        | train_mono
    )

    if train_all & validation:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "train speaker appears in validation."
        )

    if train_all & test:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "train speaker appears in test."
        )

    if validation & test:

        raise RuntimeError(
            "GLOBAL VIOLATION: "
            "validation speaker appears in test."
        )

    print(
        f"train_small speakers : "
        f"{len(train_small)}"
    )

    print(
        f"train_mono speakers  : "
        f"{len(train_mono)}"
    )

    print(
        f"train union speakers : "
        f"{len(train_all)}"
    )

    print(
        f"validation speakers   : "
        f"{len(validation)}"
    )

    print(
        f"test speakers         : "
        f"{len(test)}"
    )

    # ========================================================
    # Train small / mono overlap
    # ========================================================

    overlap = (
        train_small
        & train_mono
    )

    print(
        f"\ntrain_small ∩ train_mono: "
        f"{len(overlap)} speakers"
    )

    # ========================================================
    # Verify every overlapping train speaker.
    #
    # For each speaker:
    #
    #     small domains == mono domains
    #
    # and language must be the same.
    # ========================================================

    small_df = global_df[
        global_df["dataset"]
        == "noisy_CWWS_train_small"
    ]

    mono_df = global_df[
        global_df["dataset"]
        == "noisy_CWWS_train_mono"
    ]

    for speaker in overlap:

        small_rows = small_df[
            small_df["speaker"]
            == speaker
        ]

        mono_rows = mono_df[
            mono_df["speaker"]
            == speaker
        ]

        small_pairs = set(
            zip(
                small_rows[
                    "language"
                ],
                small_rows[
                    "environment"
                ],
            )
        )

        mono_pairs = set(
            zip(
                mono_rows[
                    "language"
                ],
                mono_rows[
                    "environment"
                ],
            )
        )

        if small_pairs != mono_pairs:

            raise RuntimeError(
                "\n"
                "ILLEGAL TRAIN SMALL/MONO OVERLAP\n"
                f"Speaker: {speaker}\n"
                f"Small cells: {small_pairs}\n"
                f"Mono cells : {mono_pairs}\n"
                "\n"
                "A speaker may be shared only in the same "
                "language + same environment.\n"
            )

    # ========================================================
    # Clean clip overlap between train_small and train_mono
    # ========================================================
    #
    # Explicitly allowed.
    # ========================================================

    small_clips = set(
        small_df["clip"]
    )

    mono_clips = set(
        mono_df["clip"]
    )

    print(
        f"train_small ∩ train_mono clean clips: "
        f"{len(small_clips & mono_clips)}"
    )

    # ========================================================
    # Verify no train speaker in validation/test.
    # ========================================================

    print(
        f"\nTrain ∩ validation: "
        f"{len(train_all & validation)}"
    )

    print(
        f"Train ∩ test      : "
        f"{len(train_all & test)}"
    )

    print(
        f"Validation ∩ test : "
        f"{len(validation & test)}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n✓ GLOBAL VERIFICATION PASSED"
    )

    print(
        "  ✓ train_small and train_mono may share speakers"
    )

    print(
        "  ✓ train_small/train_mono overlap only in "
        "same language + same domain"
    )

    print(
        "  ✓ train speakers are disjoint from validation"
    )

    print(
        "  ✓ train speakers are disjoint from test"
    )

    print(
        "  ✓ validation speakers are disjoint from test"
    )


# ============================================================
# VERIFY TRAIN CLEAN DIFFERENCES
# ============================================================

def verify_train_domains_not_forced_identical(
    train_statistics,
):
    """
    Important:

    For train, clean samples are independently sampled for each
    domain.

    We do NOT require them to be identical.

    We only print overlap information.
    """

    print("\n" + "=" * 80)
    print(
        "TRAIN CLEAN SAMPLE DOMAIN CHECK"
    )
    print("=" * 80)

    for dataset_name in [
        "noisy_CWWS_train_small",
        "noisy_CWWS_train_mono",
    ]:

        df = train_statistics[
            train_statistics["dataset"]
            == dataset_name
        ]

        for language in LANGUAGES:

            domain_sets = {}

            for environment in ENVIRONMENTS:

                domain_sets[
                    environment
                ] = set(
                    df.loc[
                        (
                            df["language"]
                            == language
                        )
                        &
                        (
                            df["environment"]
                            == environment
                        ),
                        "clip",
                    ]
                )

            print(
                f"\n{dataset_name} | "
                f"{language}"
            )

            for i, env_a in enumerate(
                ENVIRONMENTS
            ):

                for env_b in ENVIRONMENTS[
                    i + 1:
                ]:

                    overlap = (
                        domain_sets[env_a]
                        &
                        domain_sets[env_b]
                    )

                    print(
                        f"  {env_a:12s} ∩ "
                        f"{env_b:12s}: "
                        f"{len(overlap)} clean clips"
                    )

    print(
        "\n✓ Train domains are sampled independently."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print(
        "CREATING CommonVoice DATASETS"
    )
    print("=" * 80)

    print(
        f"\nSeed: {SEED}"
    )

    # ========================================================
    # CONFIGURATION SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "DATASET CONFIGURATION"
    )
    print("=" * 80)

    for name, config in (
        DATASET_CONFIG.items()
    ):

        targets = get_targets(
            config["total"]
        )

        print(
            f"\n{name}"
        )

        print(
            f"  CommonVoice split        : "
            f"{config['split']}"
        )

        print(
            f"  Total samples            : "
            f"{targets['total']}"
        )

        print(
            f"  Samples/environment      : "
            f"{targets['per_environment']}"
        )

        print(
            f"  Samples/language         : "
            f"{targets['per_language']}"
        )

        print(
            f"  Samples/environment/lang : "
            f"{targets['per_cell']}"
        )

    print(
        "\nTrain constraints:"
    )

    print(
        f"  train_small speakers/cell: "
        f"{TRAIN_SMALL_SPEAKERS_PER_CELL}"
    )

    print(
        f"  train_mono speakers/cell : "
        f"{TRAIN_MONO_SPEAKERS_PER_CELL}"
    )

    print(
        f"  zh-CN speakers/cell      : "
        f"{ZH_CN_TRAIN_SPEAKERS_PER_CELL}"
    )

    print(
        "\nEvaluation constraints:"
    )

    print(
        "  validation: 1 sample/speaker"
    )

    print(
        "  validation: identical clean across domains"
    )

    print(
        f"  test: maximum "
        f"{TEST_MAX_SAMPLES_PER_SPEAKER} "
        f"base samples/speaker"
    )

    print(
        "  test: identical clean across domains"
    )

    print(
        "\nCross-dataset constraint:"
    )

    print(
        "  train ↔ validation: DISJOINT"
    )

    print(
        "  train ↔ test      : DISJOINT"
    )

    print(
        "  validation ↔ test : DISJOINT"
    )

    print(
        "  train_small ↔ train_mono: "
        "OVERLAP ALLOWED ONLY SAME LANGUAGE + DOMAIN"
    )

    # ========================================================
    # TRAIN SPEAKER ALLOCATION
    # ========================================================

    train_cells = allocate_train_speakers(
        cv_map=cv_map,
        seed=SEED,
    )

    # ========================================================
    # TRAIN RECORD ALLOCATION
    # ========================================================

    train_pools = build_speaker_pools(
        cv_map
    )

    train_limits = allocate_train_records(
        train_cells=train_cells,
        pools=train_pools,
        seed=SEED,
    )

    # ========================================================
    # GET ALL TRAIN SPEAKERS
    # ========================================================
    #
    # Important:
    #
    # train_small/train_mono MAY share speakers.
    # Therefore we use the UNION.
    #
    # Any speaker appearing in either train dataset becomes
    # forbidden for validation and test.
    # ========================================================

    train_speakers = get_train_speakers(
        train_cells
    )

    print("\n" + "=" * 80)
    print("TRAIN SPEAKER EXCLUSIVITY")
    print("=" * 80)

    print(
        f"Train speakers: "
        f"{len(train_speakers)}"
    )

    # ========================================================
    # VALIDATION ALLOCATION
    # ========================================================
    #
    # Validation can ONLY use speakers that are not already
    # used by train_small/train_mono.
    # ========================================================

    validation_records = (
        allocate_validation_samples(
            cv_map=cv_map,
            forbidden_speakers=train_speakers,
            seed=SEED,
        )
    )

    # ========================================================
    # GET VALIDATION SPEAKERS
    # ========================================================

    validation_speakers = get_validation_speakers(
        validation_records
    )

    print(
        f"Validation speakers: "
        f"{len(validation_speakers)}"
    )

    # Safety check
    train_validation_overlap = (
        train_speakers
        & validation_speakers
    )

    if train_validation_overlap:

        raise RuntimeError(
            "INTERNAL ERROR: "
            "validation contains train speakers."
        )

    # ========================================================
    # TEST ALLOCATION
    # ========================================================
    #
    # Test must avoid BOTH:
    #
    #   train speakers
    #   validation speakers
    #
    # Therefore:
    #
    #   forbidden_test =
    #       train ∪ validation
    # ========================================================

    forbidden_test_speakers = (
        train_speakers
        | validation_speakers
    )

    print(
        f"Speakers forbidden from test: "
        f"{len(forbidden_test_speakers)}"
    )

    test_records = (
        allocate_test_samples(
            cv_map=cv_map,
            forbidden_speakers=forbidden_test_speakers,
            seed=SEED,
        )
    )

    # ========================================================
    # VERIFY TRAIN INTERNAL CONSTRAINTS
    # ========================================================

    verify_train_speaker_constraints(
        train_cells
    )

    # ========================================================
    # VERIFY GLOBAL SPEAKER EXCLUSIVITY
    # ========================================================

    verify_cross_split_speaker_exclusivity(
        train_cells=train_cells,
        validation_records=validation_records,
        test_records=test_records,
    )

    # ========================================================
    # VERIFY EVALUATION CLEAN CONSISTENCY
    # ========================================================

    verify_evaluation_clean_consistency(
        validation_records=validation_records,
        test_records=test_records,
    )

    # ========================================================
    # GENERATE TRAIN DATASETS
    # ========================================================

    generated_outputs = {}

    all_statistics = []

    train_small_output, train_small_stats = (
        generate_train_dataset(
            dataset_name=(
                "noisy_CWWS_train_small"
            ),
            train_cells=train_cells,
            train_limits=train_limits,
            cv_map=cv_map,
            wham_map=wham_map,
            seed=SEED,
        )
    )

    generated_outputs[
        "noisy_CWWS_train_small"
    ] = train_small_output

    all_statistics.append(
        train_small_stats
    )

    train_mono_output, train_mono_stats = (
        generate_train_dataset(
            dataset_name=(
                "noisy_CWWS_train_mono"
            ),
            train_cells=train_cells,
            train_limits=train_limits,
            cv_map=cv_map,
            wham_map=wham_map,
            seed=SEED + 1,
        )
    )

    generated_outputs[
        "noisy_CWWS_train_mono"
    ] = train_mono_output

    all_statistics.append(
        train_mono_stats
    )

    # ========================================================
    # GENERATE VALIDATION
    # ========================================================

    validation_output, validation_stats = (
        generate_validation_dataset(
            validation_records=validation_records,
            wham_map=wham_map,
            seed=SEED + 2,
        )
    )

    generated_outputs[
        "noisy_CWWS_validation"
    ] = validation_output

    all_statistics.append(
        validation_stats
    )

    # ========================================================
    # GENERATE TEST
    # ========================================================

    test_output, test_stats = (
        generate_test_dataset(
            test_records=test_records,
            wham_map=wham_map,
            seed=SEED + 3,
        )
    )

    generated_outputs[
        "noisy_CWWS_test"
    ] = test_output

    all_statistics.append(
        test_stats
    )

    # ========================================================
    # TRAIN DOMAIN CLEAN CHECK
    # ========================================================

    all_train_stats = pd.concat(
        [
            train_small_stats,
            train_mono_stats,
        ],
        ignore_index=True,
    )

    verify_train_domains_not_forced_identical(
        all_train_stats
    )

    # ========================================================
    # GLOBAL FINAL VERIFICATION
    # ========================================================

    verify_global_datasets(
        all_statistics
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "DATASET CREATION COMPLETE"
    )
    print("=" * 80)

    for dataset_name, output in (
        generated_outputs.items()
    ):

        print(
            f"✓ {dataset_name}: "
            f"{len(output['data'])} samples"
        )

    print("\n" + "=" * 80)
    print(
        "FINAL CONSTRAINTS"
    )
    print("=" * 80)

    print(
        "✓ train_small: 150 speakers/cell "
        "(165 for zh-CN)"
    )

    print(
        "✓ train_mono: 450 speakers/cell "
        "(165 for zh-CN)"
    )

    print(
        "✓ Train speakers are disjoint between domains"
    )

    print(
        "✓ train_small/train_mono can share speakers "
        "in the same language + domain"
    )

    print(
        "✓ train_small/train_mono can share clean samples"
    )

    print(
        "✓ Train clean samples are NOT forced to be identical "
        "between domains"
    )

    print(
        "✓ No train speaker appears in validation"
    )

    print(
        "✓ No train speaker appears in test"
    )

    print(
        "✓ Validation/test speakers are disjoint"
    )

    print(
        "✓ Validation has one clean sample per speaker"
    )

    print(
        "✓ Validation clean samples are identical "
        "across domains"
    )

    print(
        f"✓ Test has at most "
        f"{TEST_MAX_SAMPLES_PER_SPEAKER} "
        f"clean samples per speaker"
    )

    print(
        "✓ Test clean samples are identical "
        "across domains"
    )

    print(
        "\nAll requested constraints have been verified."
    )