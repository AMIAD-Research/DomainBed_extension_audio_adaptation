import argparse
import json
import re
from pathlib import Path
import os
import gc

import soundfile as sf
import torch
import numpy as np
import psutil

# from domainbed.scripts_audio.audio_classes import Resample, Normalize
# from domainbed.scripts_audio.sc_wind_noise_generator import WindNoiseGenerator as wng
# from domainbed.scripts_audio.preprocess_mixtures import (
    # load_audio,
    # mix_at_snr,
    # add_saturation,
    # get_simulated_wind,
# )

from domainbed.prepare_data.utils_dataset import get_simulated_wind, load_audio, mix_at_snr, add_saturation, Resample, Normalize
from domainbed.prepare_data.sc_wind_noise_generator import WindNoiseGenerator as wng
process = psutil.Process(os.getpid())

def get_language(filename):
    """
    common_voice_zh-CN_19877584.mp3
        -> zh-CN
    """
    m = re.match(r"common_voice_(.+)_\d+\.[^.]+$", filename)
    if m is None:
        raise ValueError(f"Cannot extract language from {filename}")
    return m.group(1)


def get_clean_path(commonvoice_root, filename):
    language = get_language(filename)
    return Path(commonvoice_root) / language / "clips" / filename


def rebuild_dataset(
    dataset_info_path,
    commonvoice_root,
    wham_root,
    output_root,
    sample_rate=16000,
):

    commonvoice_root = Path(commonvoice_root)
    wham_root = Path(wham_root)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    with open(dataset_info_path, "r") as f:
        dataset = json.load(f)

    transforms = [
        Resample(sample_rate),
        Normalize(),
    ]

    LANGUAGES = ["fr", "de", "es", "zh-CN"]

    total = len(dataset["data"])

    for i, sample in enumerate(dataset["data"], 1):

        if i % 100 == 0:
            print(f"RSS: {process.memory_info().rss / 1024**2:.1f} MB")

        clean_relpath = Path(sample["clean"])
        clean_path = commonvoice_root / clean_relpath

        if not clean_path.exists():
            raise FileNotFoundError(clean_path)

        clean_name = clean_relpath.name
        language = clean_relpath.parts[0]

        noise = sample["noise"]
        snr = sample["snr_db"]
        gain = sample["gain"]


        with torch.no_grad():

            print(f"\n===== SAMPLE {i}/{total} =====")

            print(f"Loading clean {clean_relpath}")
            clean, sr = load_audio(clean_path, transforms)

            print(f"Generating corruption {noise}")

            ###########################################################
            # Apply corruption
            ###########################################################

            if noise == "clean":

                output = clean

            elif noise == "wind":

                wind = get_simulated_wind(gustiness, wind_profile, new_point)

                output = mix_at_snr(clean, wind, snr)

                peak = output.abs().max().clamp(min=1.0)
                output = output / peak

            elif noise == "saturation":

                output = add_saturation(clean, gain)

            elif noise.startswith("wham_noise"):

                noise_path = wham_root / Path(noise).relative_to("wham_noise")

                # if not noise_path.exists():
                #     raise FileNotFoundError(noise_path)

                if not noise_path.exists():
                    print(f"Warning: noise file not found: {noise_path}")
                    continue

                noise_signal, _ = load_audio(
                    noise_path,
                    transforms,
                )

                output = mix_at_snr(
                    clean,
                    noise_signal,
                    snr,
                )

                peak = output.abs().max().clamp(min=1.0)
                output = output / peak

                del noise_signal

            else:

                raise ValueError(f"Unknown noise type: {noise}")

        ###########################################################
        # Save using the original dataset structure
        ###########################################################

        domain = "wham" if noise.startswith("wham_noise") else noise

        label = LANGUAGES.index(language)

        out_dir = output_root / domain / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / clean_name

        print("Saving")

        sf.write(
            out_path,
            output.cpu().numpy(),
            sr,
            format="MP3",
        )

        print("Saved")

        del clean
        del output

        gc.collect()

        print(process.memory_info().rss / 1024**2)


        print(f"[{i:5d}/{total}] {out_path}")

    print("\nDataset successfully rebuilt.")


if __name__ == "__main__":

    ## To be removed ! 

    # Directory containing this script
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Project root (domainbed/)
    PROJECT_ROOT = SCRIPT_DIR.parent

    # Data directory
    DATA_DIR = PROJECT_ROOT / "data"
    json_path = DATA_DIR / "noisy_CWWS_train_small.json"
    
    path_to_data = Path(os.environ["SCRATCH"]) / "domainbed/data"
    cv_root = Path(path_to_data /"URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20")
    wham_noise_root = Path(path_to_data / "URGENT/noises/wham_noise/")   

    out_root = output_root = Path(path_to_data / f"NOISY_CWWS_TESTSET_wind_2")


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_info",
        default=json_path,
        # required=True,
        help="dataset_info_correct.json",
    )

    parser.add_argument(
        "--commonvoice_root",
        default=cv_root,
        # required=True,
        help="Root of CommonVoice (cv-corpus-22.0)",
    )

    parser.add_argument(
        "--wham_root",
        default=wham_noise_root,
        # required=True,
        help="Root directory containing wham_noise/",
    )

    parser.add_argument(
        "--output_root",
        default= out_root, 
        # required=True,
        help="Directory where rebuilt dataset will be written",
    )

    parser.add_argument(
        "--sample_rate",
        default=16000,
        type=int,
    )

    args = parser.parse_args()

    rebuild_dataset(
        dataset_info_path=args.dataset_info,
        commonvoice_root=args.commonvoice_root,
        wham_root=args.wham_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
    )