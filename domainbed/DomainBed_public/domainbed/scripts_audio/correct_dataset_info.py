import json
from pathlib import Path
import csv
from collections import Counter
import os


LANGUAGES = ["fr", "de", "es", "zh-CN"]

def compare_dataset_info(
    json1_path,
    json2_path,
    commonvoice_root,
    languages=("fr", "de", "es", "zh-CN"),
):
    """
    Compare two dataset_info.json files.

    Reports:
      - WHAM noise overlap
      - Speaker overlap
      - Noise-type statistics
      - Language statistics
    """

    # --------------------------------------------------
    # Load Common Voice metadata
    # --------------------------------------------------
    filename_to_speaker = {}
    filename_to_language = {}

    valid_tsv = {"train.tsv", "dev.tsv", "test.tsv"}

    for lang in languages:
        lang_dir = os.path.join(commonvoice_root, lang)

        for file in os.listdir(lang_dir):
            if file not in valid_tsv:
                continue

            with open(os.path.join(lang_dir, file), encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                for row in reader:
                    filename = os.path.basename(row["path"])
                    filename_to_speaker[filename] = row["client_id"]
                    filename_to_language[filename] = row["locale"]

    # --------------------------------------------------
    # Load JSON
    # --------------------------------------------------
    with open(json1_path) as f:
        ds1 = json.load(f)["data"]

    with open(json2_path) as f:
        ds2 = json.load(f)["data"]

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------
    def analyse(dataset):

        wham_files = set()
        speakers = set()

        noise_counter = Counter()
        language_counter = Counter()

        for sample in dataset:

            clean = sample["clean"]
            noise = sample["noise"]

            # ---------- language ----------
            if "fr" in clean:
                lang = "fr"
            elif "zh" in clean:
                lang = "zh-CN"
            elif "es" in clean:
                lang = "es"
            elif "de" in clean:
                lang = "de"
            else:
                lang = "Unknown"
            language_counter[lang] += 1

            # ---------- speaker ----------
            spk = filename_to_speaker.get(clean)
            if spk is not None:
                speakers.add(spk)

            # ---------- noise type ----------
            if noise is None:
                noise_type = "clean_or_other"

            elif "wham_noise" in noise:
                noise_type = "wham"
                wham_files.add(noise)
            elif "clean" in noise:
                noise_type = "clean"
            elif "satu" in noise:
                noise_type = "saturation"
            elif "wind" in noise:
                noise_type = "wind"
            else:
                noise_type = "unknown"

            noise_counter[noise_type] += 1

        return {
            "wham": wham_files,
            "speakers": speakers,
            "noise_stats": noise_counter,
            "language_stats": language_counter,
        }

    info1 = analyse(ds1)
    info2 = analyse(ds2)

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------
    common_wham = info1["wham"] & info2["wham"]
    common_speakers = info1["speakers"] & info2["speakers"]

    print("=" * 70)
    print("DATASET COMPARISON")
    print("=" * 70)

    print(f"\nShared WHAM noises : {len(common_wham)}")

    if common_wham:
        print("Examples:")
        for f in sorted(list(common_wham))[:10]:
            print("   ", f)

    print(f"\nShared speakers : {len(common_speakers)}")

    if common_speakers:
        print("Examples:")
        for s in list(common_speakers)[:10]:
            print("   ", s)

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    for name, info in zip(
        ["Dataset 1", "Dataset 2"],
        [info1, info2],
    ):

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print("\nNoise statistics")
        for k, v in sorted(info["noise_stats"].items()):
            print(f"{k:20s}: {v}")

        print("\nLanguage statistics")
        for k, v in sorted(info["language_stats"].items()):
            print(f"{k:10s}: {v}")

    return {
        "shared_wham": common_wham,
        "shared_speakers": common_speakers,
        "dataset1": info1,
        "dataset2": info2,
    }

def build_commonvoice_index(commonvoice_root, languages):
    """
    Build:
        filename -> relative path in CommonVoice

    Only for the selected languages.
    """

    commonvoice_root = Path(commonvoice_root)

    index = {}

    for language in languages:

        clips_dir = commonvoice_root / language / "clips"

        if not clips_dir.exists():
            raise FileNotFoundError(clips_dir)

        print(f"Indexing {language}...")

        n = 0

        for mp3 in clips_dir.rglob("*.mp3"):
            index[mp3.name] = str(mp3.relative_to(commonvoice_root))
            n += 1

        print(f"  {n} files")

    print(f"Indexed {len(index)} files in total.")

    return index

def build_cv_speaker_index(commonvoice_root, languages):
    """
    Returns
    -------
    dict
        Maps relative clip path (e.g. 'fr/clips/xxx.mp3')
        to Common Voice client_id.
    """
    commonvoice_root = Path(commonvoice_root)

    mapping = {}

    tsv_files = [
        "train.tsv",
        "dev.tsv",
        "test.tsv",
        "validated.tsv",
        "invalidated.tsv",
        "other.tsv",
    ]

    for lang in languages:
        lang_root = commonvoice_root / lang

        for tsv in tsv_files:
            path = lang_root / tsv
            if not path.exists():
                continue

            with open(path, newline="", encoding="utf8") as f:
                reader = csv.DictReader(f, delimiter="\t")

                for row in reader:
                    rel_path = f"{lang}/clips/{row['path']}"
                    mapping[rel_path] = row["client_id"]

    return mapping

def correct_dataset_info(input_json, commonvoice_root, output_json=None):
    """
    Replace the 'noise' field for clean/wind/saturation samples
    and replace 'clean' by its relative CommonVoice path.
    """

    input_json = Path(input_json)

    if output_json is None:
        output_json = input_json.with_name("dataset_info_correct.json")
    else:
        output_json = Path(output_json)

    # Build CV index once
    print("Building CommonVoice index...")
    cv_index = build_commonvoice_index(
        commonvoice_root,
        ["fr", "de", "es", "zh-CN"],
        )
    print("Index built.")

    with open(input_json, "r") as f:
        dataset = json.load(f)

    for sample in dataset["data"]:

        ###########################################################
        # Replace clean filename by relative CommonVoice path
        ###########################################################

        filename = sample["clean"]

        if filename not in cv_index:
            raise FileNotFoundError(filename)

        sample["clean"] = cv_index[filename]

        ###########################################################
        # Replace noise field
        ###########################################################

        noise = sample["noise"]
        snr = sample["snr_db"]
        gain = sample["gain"]

        # WHAM -> keep original path
        if noise is not None:
            continue

        # clean
        if snr is None and gain is None:
            sample["noise"] = "clean"

        # wind
        elif snr is not None and gain is None:
            sample["noise"] = "wind"

        # saturation
        elif snr is None and gain is not None:
            sample["noise"] = "saturation"

        else:
            raise ValueError(f"Unknown sample description:\n{sample}")

    print(f"Writing {output_json}...")
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved corrected file to: {output_json}")

import json
from pathlib import Path

def remove_train_speakers(
    train_dataset_path,
    test_dataset_path,
    commonvoice_root,
    output_test_dataset_path=None,
):
    train_dataset_path = Path(train_dataset_path)
    test_dataset_path = Path(test_dataset_path)

    if output_test_dataset_path is None:
        output_test_dataset_path = test_dataset_path.with_name(
            "dataset_info_no_train_speakers.json"
        )

    with open(train_dataset_path) as f:
        train_dataset = json.load(f)

    with open(test_dataset_path) as f:
        test_dataset = json.load(f)

    speaker_index = build_cv_speaker_index(
        commonvoice_root,
        ["fr", "de", "es", "zh-CN"],
    )

    # Speakers used in train
    train_speakers = {
        speaker_index[sample["clean"]]
        for sample in train_dataset["data"]
    }

    original_size = len(test_dataset["data"])

    filtered = [
        sample
        for sample in test_dataset["data"]
        if speaker_index[sample["clean"]] not in train_speakers
    ]

    removed = original_size - len(filtered)

    test_dataset["data"] = filtered

    with open(output_test_dataset_path, "w") as f:
        json.dump(test_dataset, f, indent=2)

    print(f"Train speakers       : {len(train_speakers)}")
    print(f"Original test samples: {original_size}")
    print(f"Removed samples      : {removed}")
    print(f"Remaining test       : {len(filtered)}")


def remove_train_overlap(
    train_dataset_path,
    test_dataset_path,
    output_test_dataset_path=None,
):
    """
    Remove from the test dataset every sample whose clean speech file
    is also present in the train dataset.

    Parameters
    ----------
    train_dataset_path : str or Path
        Path to the train dataset_info.json.

    test_dataset_path : str or Path
        Path to the test dataset_info.json.

    output_test_dataset_path : str or Path, optional
        Output path. If None, creates
        'dataset_info_no_overlap.json' next to the test json.
    """

    train_dataset_path = Path(train_dataset_path)
    test_dataset_path = Path(test_dataset_path)

    if output_test_dataset_path is None:
        output_test_dataset_path = test_dataset_path.with_name(
            "dataset_info_no_overlap.json"
        )
    else:
        output_test_dataset_path = Path(output_test_dataset_path)

    # Load datasets
    with open(train_dataset_path, "r") as f:
        train_dataset = json.load(f)

    with open(test_dataset_path, "r") as f:
        test_dataset = json.load(f)

    # Set of all clean files used in training
    train_clean_files = {
        sample["clean"]
        for sample in train_dataset["data"]
    }

    original_size = len(test_dataset["data"])

    # Keep only samples not present in train
    filtered_data = [
        sample
        for sample in test_dataset["data"]
        if sample["clean"] not in train_clean_files
    ]

    removed = original_size - len(filtered_data)

    test_dataset["data"] = filtered_data

    # Save
    with open(output_test_dataset_path, "w") as f:
        json.dump(test_dataset, f, indent=2)

    print(f"Train samples        : {len(train_dataset['data'])}")
    print(f"Original test samples: {original_size}")
    print(f"Removed overlaps     : {removed}")
    print(f"Remaining test       : {len(filtered_data)}")
    print(f"Saved to             : {output_test_dataset_path}")


if __name__ == "__main__":

    dataset_path = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/Y_NOISY_Y_NEW_CWWS_TESTSET/dataset_info.json"   

    # print(dataset_path)
    commonvoice_root = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20"

    

    
    # breakpoint()
    correct_dataset_info(
        input_json=dataset_path,
        commonvoice_root=commonvoice_root,
    )

    # Test dataset
    dataset_path = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/Y_NOISY_Y_NEW_CWWS_TESTSET/dataset_info_correct.json" 

    # train dataset
    train_dataset_path = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/NOISY_AUDIO_DA_CWWS/dataset_info_correct.json"
    
    new_out = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/data/Y_NOISY_Y_NEW_CWWS_TESTSET/no_speaker_overlap_dataset_info.json"

    # breakpoint()
    # remove_train_overlap(train_dataset_path, dataset_path, new_out)


    # remove_train_speakers(train_dataset_path, dataset_path, commonvoice_root, new_out)

    dict = compare_dataset_info(dataset_path, train_dataset_path, commonvoice_root)

    breakpoint()

    