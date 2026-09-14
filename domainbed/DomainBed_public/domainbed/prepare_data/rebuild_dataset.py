import argparse
import json
import re
from pathlib import Path
import os
import gc
import shutil

import soundfile as sf
import torch
import numpy as np
import psutil
import time

from domainbed.prepare_data.utils_dataset import (
    get_simulated_wind,
    load_audio,
    mix_at_snr,
    add_saturation,
    Resample,
    Normalize,
)

process = psutil.Process(os.getpid())


# ============================================================
# UTILITIES
# ============================================================

def get_language(filename):
    """
    Example:
        common_voice_zh-CN_19877584.mp3
            -> zh-CN
    """

    m = re.match(
        r"common_voice_(.+)_\d+\.[^.]+$",
        filename,
    )

    if m is None:
        raise ValueError(
            f"Cannot extract language from {filename}"
        )

    return m.group(1)


def get_clean_path(commonvoice_root, clean_relpath):
    """
    JSON example:

        zh-CN/clips/00/common_voice_zh-CN_123.mp3

    becomes:

        commonvoice_root/
            zh-CN/
                clips/
                    00/
                        common_voice_zh-CN_123.mp3
    """

    clean_relpath = Path(clean_relpath)

    if clean_relpath.is_absolute():
        return clean_relpath

    return Path(commonvoice_root) / clean_relpath


def get_wham_path(wham_root, noise_path):

    noise_path = Path(noise_path)

    if noise_path.is_absolute():

        if noise_path.exists():
            return noise_path

        parts = noise_path.parts

        if "wham_noise" in parts:

            idx = parts.index("wham_noise")

            relative_part = Path(
                *parts[idx + 1:]
            )

            return Path(wham_root) / relative_part

        return noise_path

    if noise_path.parts[0] == "wham_noise":

        return (
            Path(wham_root)
            / Path(*noise_path.parts[1:])
        )

    return Path(wham_root) / noise_path

# ============================================================
# VALIDATE JSON
# ============================================================

def validate_dataset_json(dataset):
    """
    Validate that the JSON contains all fields required
    for reconstruction.
    """

    required_sampler_fields = [
        "wind_gustiness_range",
        "wind_profile_magnitude_range",
        "wind_snr_range",
        "wham_snr_range",
        "saturation_gain_range",
        "distribution",
    ]

    if "sampler_info" not in dataset:

        raise ValueError(
            "JSON does not contain 'sampler_info'."
        )

    if "data" not in dataset:

        raise ValueError(
            "JSON does not contain 'data'."
        )

    sampler_info = dataset["sampler_info"]

    missing = [
        field
        for field in required_sampler_fields
        if field not in sampler_info
    ]

    if missing:

        raise ValueError(
            "Missing fields in sampler_info: "
            f"{missing}"
        )

    required_sample_fields = [
        "clean",
        "noise",
        "snr_db",
        "gain",
        "gustiness",
        "wind_profile",
    ]

    for i, sample in enumerate(
        dataset["data"]
    ):

        missing = [
            field
            for field in required_sample_fields
            if field not in sample
        ]

        if missing:

            raise ValueError(
                f"Sample {i} is missing fields: "
                f"{missing}"
            )


# ============================================================
# DATASET REBUILD
# ============================================================

def rebuild_dataset(
    dataset_info_path,
    commonvoice_root,
    wham_root,
    output_root=None,
    sample_rate=16000,
    overwrite=False,
    debug=False
):

    dataset_info_path = Path(
        dataset_info_path
    )

    commonvoice_root = Path(
        commonvoice_root
    )

    wham_root = Path(
        wham_root
    )

    # --------------------------------------------------------
    # Automatically generate output_root
    # from JSON filename
    # --------------------------------------------------------

    if output_root is None:

        dataset_name = (
            dataset_info_path
            .stem
        )

        path_to_data = (
            Path(os.environ["SCRATCH"])
            / "domainbed/data"
        )

        output_root = (
            path_to_data
            / dataset_name
        )

    else:

        output_root = Path(
            output_root
        )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOADING DATASET JSON")
    print("=" * 80)

    print(
        f"JSON       : {dataset_info_path}"
    )

    print(
        f"Output     : {output_root}"
    )

    with open(
        dataset_info_path,
        "r",
        encoding="utf-8",
    ) as f:

        dataset = json.load(f)

    validate_dataset_json(
        dataset
    )

    sampler_info = dataset[
        "sampler_info"
    ]

    samples = dataset[
        "data"
    ]

    total = len(samples)

    print(
        f"Samples    : {total}"
    )

    print(
        f"Audio ext  : "
        f"{dataset.get('audio_ext', '.mp3')}"
    )

    print(
        f"Distribution: "
        f"{sampler_info['distribution']}"
    )

    # --------------------------------------------------------
    # Print ranges actually stored in JSON
    # --------------------------------------------------------

    print("\nSampler configuration:")

    print(
        "  wind gustiness range :",
        sampler_info[
            "wind_gustiness_range"
        ],
    )

    print(
        "  wind profile range   :",
        sampler_info[
            "wind_profile_magnitude_range"
        ],
    )

    print(
        "  wind SNR range       :",
        sampler_info[
            "wind_snr_range"
        ],
    )

    print(
        "  WHAM SNR range       :",
        sampler_info[
            "wham_snr_range"
        ],
    )

    print(
        "  saturation gain      :",
        sampler_info[
            "saturation_gain_range"
        ],
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if output_root.exists():

        if overwrite:

            print(
                f"\nRemoving existing output: "
                f"{output_root}"
            )

            shutil.rmtree(
                output_root
            )

        else:

            print(
                f"\nOutput directory already exists:"
                f"\n{output_root}"
            )

            print(
                "Existing files will be overwritten "
                "individually."
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Audio transforms
    # --------------------------------------------------------

    transforms = [
        Resample(sample_rate),
        Normalize(),
    ]

    LANGUAGES = [
        "fr",
        "de",
        "es",
        "zh-CN",
    ]

    # ========================================================
    # STATISTICS
    # ========================================================

    statistics = {
        "clean": 0,
        "wham": 0,
        "wind": 0,
        "saturation": 0,
    }

    language_statistics = {
        language: 0
        for language in LANGUAGES
    }

    if debug:
        debug_counts = {
            domain: {language: 0 for language in LANGUAGES}
            for domain in statistics
        }

    # ========================================================
    # PROCESS SAMPLES
    # ========================================================

    for i, sample in enumerate(
        samples,
        1,
    ):
        if debug and all(
            debug_counts[domain][language] >= 3
            for domain in debug_counts
            for language in debug_counts[domain]
        ):
            break

        if i % 100 == 0:

            print(
                f"\nRSS: "
                f"{process.memory_info().rss / 1024**2:.1f} MB"
            )

        print(
            f"\n===== SAMPLE {i}/{total} ====="
        )

        # ----------------------------------------------------
        # Read JSON fields
        # ----------------------------------------------------

        clean_relpath = Path(
            sample["clean"]
        )

        noise = sample[
            "noise"
        ]

        snr = sample[
            "snr_db"
        ]

        gain = sample[
            "gain"
        ]

        gustiness = sample[
            "gustiness"
        ]

        wind_profile = [sample[
            "wind_profile"
        ]]

        if noise == "clean":
            expected_domain = "clean"
        elif noise == "wind":
            expected_domain = "wind"
        elif noise == "saturation":
            expected_domain = "saturation"
        elif isinstance(noise, str) and noise.startswith("wham_noise"):
            expected_domain = "wham"
        else:
            expected_domain = None
        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        if len(
            clean_relpath.parts
        ) == 0:

            raise ValueError(
                f"Invalid clean path: "
                f"{clean_relpath}"
            )

        language = (
            clean_relpath.parts[0]
        )

        if debug and expected_domain is not None:
            if debug_counts[expected_domain][language] >= 3:
                continue

        if language not in LANGUAGES:

            raise ValueError(
                f"Unknown language "
                f"'{language}' in "
                f"{clean_relpath}"
            )

        # ----------------------------------------------------
        # Clean path
        # ----------------------------------------------------

        clean_path = get_clean_path(
            commonvoice_root,
            clean_relpath,
        )

        if not clean_path.exists():

            raise FileNotFoundError(
                "\nClean audio not found:\n"
                f"  JSON path : {clean_relpath}\n"
                f"  Resolved  : {clean_path}"
            )

        clean_name = (
            clean_relpath.name
        )

        print(
            f"Clean     : {clean_relpath}"
        )

        print(
            f"Noise     : {noise}"
        )

        # ====================================================
        # LOAD CLEAN
        # ====================================================

        with torch.no_grad():

            clean, sr = load_audio(
                clean_path,
                transforms,
            )

            # =================================================
            # CLEAN
            # =================================================

            if noise == "clean":

                print(
                    "Corruption: clean"
                )

                output = clean

                domain = "clean"

            # =================================================
            # WIND
            # =================================================

            elif noise == "wind":

                print("Corruption: wind")

                print(f"  gustiness    = {gustiness}")
                print(f"  wind_profile = {wind_profile}")
                print(f"  snr_db       = {snr}")

                print(">>> Generating wind...")

                wind = get_simulated_wind(
                    gustiness,
                    wind_profile,
                )


                print(
                    f">>> clean shape: {clean.shape}"
                )

                print(
                    f">>> wind shape: {wind.shape}"
                )

                print(
                    f">>> clean dtype: {clean.dtype}"
                )

                print(
                    f">>> wind dtype: {wind.dtype}"
                )

                print(">>> Mixing at SNR...")

                output = mix_at_snr(
                    clean,
                    wind,
                    snr,
                )


                peak = (
                    output.abs()
                    .max()
                    .clamp(min=1.0)
                )

                output = output / peak

                del wind

                domain = "wind"

            # =================================================
            # SATURATION
            # =================================================

            elif noise == "saturation":

                print(
                    "Corruption: saturation"
                )

                print(
                    f"  gain = {gain}"
                )

                if gain is None:

                    raise ValueError(
                        f"Missing gain "
                        f"for saturation "
                        f"sample {i}"
                    )

                output = add_saturation(
                    clean,
                    gain,
                )

                domain = "saturation"

            # =================================================
            # WHAM
            # =================================================

            elif (
                isinstance(noise, str)
                and noise.startswith(
                    "wham_noise"
                )
            ):

                print(
                    "Corruption: WHAM"
                )

                print(
                    f"  SNR = {snr}"
                )

                if snr is None:

                    raise ValueError(
                        f"Missing SNR "
                        f"for WHAM sample {i}"
                    )

                # ------------------------------------------------
                # Resolve WHAM path
                # ------------------------------------------------

                noise_path = get_wham_path(
                    wham_root,
                    noise,
                )

                print(
                    f"WHAM path: "
                    f"{noise_path}"
                )

                if not noise_path.exists():

                    raise FileNotFoundError(
                        "\nWHAM noise not found:\n"
                        f"  JSON path : {noise}\n"
                        f"  Resolved  : {noise_path}"
                    )

                noise_signal, _ = load_audio(
                    noise_path,
                    transforms,
                )

                output = mix_at_snr(
                    clean,
                    noise_signal,
                    snr,
                )

                peak = (
                    output.abs()
                    .max()
                    .clamp(min=1.0)
                )

                output = (
                    output / peak
                )

                del noise_signal

                domain = "wham"

            # =================================================
            # UNKNOWN
            # =================================================

            else:

                raise ValueError(
                    f"Unknown noise type: "
                    f"{noise}"
                )

        # ====================================================
        # UPDATE STATISTICS
        # ====================================================

        statistics[
            domain
        ] += 1

        language_statistics[
            language
        ] += 1

        
        if debug:
            debug_counts[domain][language] += 1       

        # ====================================================
        # OUTPUT STRUCTURE
        #
        # output_root/
        #     clean/
        #         0/
        #         1/
        #         2/
        #         3/
        #
        #     wham/
        #         0/
        #         ...
        #
        #     wind/
        #         ...
        #
        #     saturation/
        #         ...
        #
        # Labels:
        #   0 = fr
        #   1 = de
        #   2 = es
        #   3 = zh-CN
        # ====================================================

        label = LANGUAGES.index(
            language
        )

        out_dir = (
            output_root
            / domain
            / str(label)
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_path = (
            out_dir
            / clean_name
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        print(
            f"Saving -> {out_path}"
        )

        output_np = output.detach().cpu().numpy()

        sf.write(
            out_path,
            output_np,
            sr,
            format="MP3",
        )

        print(
            "Saved"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        # Explicitly release NumPy array
        del output_np

        # Explicitly release tensors
        del output
        del clean

        # Release any remaining temporary objects
        gc.collect()

        # If CUDA is used anywhere
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"RSS: "
            f"{process.memory_info().rss / 1024**2:.1f} MB"
        )

        print(
            f"[{i:5d}/{total}] "
            f"{out_path}"
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print("\n" + "=" * 80)
    print("DATASET REBUILD COMPLETE")
    print("=" * 80)

    print(
        f"Output root: {output_root}"
    )

    print(
        f"Total samples: {total}"
    )

    print("\nSamples per domain:")

    for domain, count in statistics.items():

        print(
            f"  {domain:12s}: {count}"
        )

    print("\nSamples per language:")

    for language, count in (
        language_statistics.items()
    ):

        print(
            f"  {language:8s}: {count}"
        )

    print(
        "\nDataset successfully rebuilt."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild an audio dataset from "
            "a generated JSON description."
        )
    )

    parser.add_argument(
        "--dataset_info",
        required=True,
        help=(
            "Path to the generated .json file."
        ),
    )

    parser.add_argument(
        "--commonvoice_root",
        default=(
            Path(os.environ["SCRATCH"])
            / "domainbed/data"
            / "URGENT/corpus/CommonVoice/"
            "cv-corpus-22.0-2025-06-20"
        ),
        help=(
            "Root of CommonVoice "
            "(cv-corpus-22.0)."
        ),
    )

    parser.add_argument(
        "--path_to_wham_noise",
        default=(
            Path(os.environ["SCRATCH"])
            / "domainbed/data"
            / "URGENT/noises/wham_noise"
        ),
        help=(
            "Root directory containing "
            "WHAM noise."
        ),
    )

    parser.add_argument(
        "--output_root",
        default=None,
        required=True,
        help=(
            "Output directory. If omitted, "
            "it is automatically generated as "
            "$SCRATCH/domainbed/data/<JSON_stem>/"
        ),
    )

    parser.add_argument(
        "--sample_rate",
        default=16000,
        type=int,
        help="Output sample rate.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete the existing output directory "
            "before rebuilding."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Delete the existing output directory "
            "before rebuilding."
        ),
    )

    args = parser.parse_args()

    rebuild_dataset(
        dataset_info_path=args.dataset_info,
        commonvoice_root=args.commonvoice_root,
        wham_root=args.path_to_wham_noise,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        debug=args.debug,
    )