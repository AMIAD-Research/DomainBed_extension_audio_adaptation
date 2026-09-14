import numpy as np 
import random as rd 
import os
import soundfile as sf
import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json 
import shutil
import sys 
from domainbed.scripts_audio.audio_classes import Resample, Normalize
from domainbed.lib import misc
from collections import defaultdict
import torchaudio
from domainbed.scripts_audio.sc_wind_noise_generator import WindNoiseGenerator as wng


def get_subpath_from_folder(full_path, folder_name="wham_noise"):
    """
    Returns the part of full_path starting from folder_name.
    Example:
        /data/datasets/wham_noise/tr/file.wav
        -> wham_noise/tr/file.wav
    """
    if full_path == None: 
        return None
    else: 
        # Normalize separators for cross-platform
        parts = os.path.normpath(full_path).split(os.sep)
        if folder_name in parts:
            idx = parts.index(folder_name)
            return os.path.join(*parts[idx:])
        else:
            # fallback: just return the filename
            return os.path.basename(full_path)

def log_noisy_sample(
    log_file,
    clean_data,
    noise_path,
    snr_db,
    gain,
    distribution,
    wind_snr_range,
    wham_snr_range,
    saturation_gain_range,
    audio_ext, 
):
    """
    Log information about a generated noisy sample in a JSON file.

    First call (clean_data=None) creates the file with top-level info:
        - snr_sampler_info
        - audio_ext
        - data: []
    Subsequent calls append each sample under "data".
    """
    sample_dict = None  # initialize

    # --- First call: initialize JSON ---
    if clean_data is None:
        
        json_data = {
            "sampler_info": {
                "wind_snr_range": wind_snr_range,
                "wham_snr_range": wham_snr_range,
                "saturation_gain_range": saturation_gain_range,
                "distribution": distribution
            },
            "audio_ext": audio_ext,
            "data": []
        }
        with open(log_file, "w") as f:
            json.dump(json_data, f, indent=2)
        return  # nothing more to do

    # --- Normal call: create sample entry ---
    
    sample_dict = {
        "clean": clean_data,
        "noise": get_subpath_from_folder(noise_path, folder_name="wham_noise"),
        "snr_db": snr_db,
        "gain": gain,
    }

    # --- Load existing JSON safely ---
    with open(log_file, "r") as f:
        json_data = json.load(f)

    # --- Append sample only if it exists ---
    if sample_dict is not None:
        json_data.setdefault("data", []).append(sample_dict)

    # --- Write back ---
    with open(log_file, "w") as f:
        json.dump(json_data, f, indent=2)



def triangular_distribution(a=-20.0, b=0.0, c=20.0):
    """ Sample from a triangular distribution 

             b
            /\\
           /  \\
          /    \\
         a      c
    """
    assert a < b and b < c, f"Contraint {a} < {b} < {c}"

    y = rd.random()

    if y < (b-a) / (c-a): 
        # x in [a, b]
        return a + np.sqrt(y * (b-a) * (c-a))
    # x in [b, c]
    return c + np.sqrt((1-y) * (c-b) * (c-a))

def uniform_distribution(min=-20, max=20): 
    return rd.uniform(min, max)

def mix_at_snr(clean, noise, snr_db):
    """
    clean, noise: torch.Tensor [T]
    """

    noise = noise.squeeze()
    clean = clean.squeeze()

    if noise.numel() < clean.numel():
        repeats = (clean.numel() + noise.numel() - 1) // noise.numel()
        noise = noise.repeat(repeats)
     
    noise = noise[:clean.numel()]
    assert clean.shape == noise.shape, f"noise shape {noise.shape} and clean shape {clean.shape} are not the same"


    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (snr_linear * noise_power + 1e-12))

    return clean + scale * noise


def add_saturation(x, gain):
    """
    Works for [T] or [B,1,T]
    """

    if x.dim() == 1:
        rms = torch.sqrt(torch.mean(x**2))
        x_norm = x / (rms + 1e-8)
    else:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        x_norm = x / (rms + 1e-8)

    x_amp = gain * x_norm
    sat_x = torch.clamp(x_amp, -1.0, 1.0)

    return sat_x

def load_audio(
    path,
    transforms=None,   # e.g. [Resample(16000), Normalize()]
):
    """
    Load audio with soundfile and apply torch-based transforms.
    Returns: (waveform, sr) where waveform is 1D torch.Tensor
    """
    
    # Load with soundfile
    audio, sr = sf.read(path, dtype="float32")
    # assert sr == sample_rate, f"Expected {sample_rate}, got {sr}"

    # Convert to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # To torch tensor (shape: [T])
    waveform = torch.from_numpy(audio)

    data = (waveform, sr)

    # Apply transforms sequentially
    if transforms is not None:
        for t in transforms:
            data = t(data)

    return data

def get_snr_sampler(
    distribution,
    snr_range,
):
    """
    distribution: ["uniform", "triangular"]
    returns: callable snr_sampler()
    """
    if distribution == "uniform":
        return lambda: uniform_distribution(snr_range[0], snr_range[2])
    if distribution == "triangular":
        return lambda: triangular_distribution(a=snr_range[0], b=snr_range[1], c=snr_range[2])

def get_gain_sampler(
    distribution,
    gain_range,
):
    """
    distribution: ["uniform", "triangular"]
    returns: callable gain_sampler()
    """
    if distribution == "uniform":
        return lambda: uniform_distribution(gain_range[0], gain_range[2])
    if distribution == "triangular":
        return lambda: triangular_distribution(a=gain_range[0], b=gain_range[1], c=gain_range[2])

def get_noise_type_from_root(root, noise_types):
    parts = Path(root).parts
    for noise in noise_types:
        if noise in parts:
            return noise
    raise ValueError(f"No noise type found in path: {root}")

def get_simulated_wind():
    gustiness_range = [8, 11]  # highly turbulent
    gustiness = np.random.uniform(gustiness_range[0], gustiness_range[1])
    number_points_wind_profile = int(1.5 * gustiness)

    # 2. Random wind profile points
    wind_profile_magnitude_range = [8, 15]
    wind_profile_acceptable_transition_threshold = 1.5

    wind_profile = [
        np.random.uniform(
            wind_profile_magnitude_range[0], wind_profile_magnitude_range[1]
        )
    ]

    while len(wind_profile) < number_points_wind_profile:
        is_valid = False
        while not is_valid:
            new_point = np.random.uniform(
                wind_profile_magnitude_range[0], wind_profile_magnitude_range[1]
            )
            is_valid = (
                new_point
                < wind_profile[-1] + wind_profile_acceptable_transition_threshold
                and new_point
                > wind_profile[-1] - wind_profile_acceptable_transition_threshold
            )
        wind_profile.append(new_point)

    # 3. Generate wind noise using the simulator
    wn_instance = wng(
        fs=16000,
        duration=3,
        generate=True,
        wind_profile=wind_profile,
        gustiness=gustiness,
        start_seed=None,  # optional: could randomize per sample
    )
    wn_signal_np, _ = wn_instance.generate_wind_noise()

    # 4. Convert to torch.Tensor and shape [1, 1, time]
    wn_signal_tensor = torch.from_numpy(wn_signal_np).float().unsqueeze(0).unsqueeze(0)
    return wn_signal_tensor

def generate_noisy_dataset(
    targeted_domain,
    speech_root,
    noise_root,
    output_root,
    distribution,
    wind_snr_range,
    wham_snr_range,
    saturation_gain_range,
    sample_rate=16000,
    audio_ext=".mp3"
):
    """
    speech_root: root folder of clean speech (tree structure preserved)
    noise_root: folder containing noise files
    output_root: where to write noised speech
    snr_sampler: function returning an SNR in dB
    """

    snr_sampler_wind = get_snr_sampler(distribution=distribution, snr_range=wind_snr_range)
    snr_sampler_wham = get_snr_sampler(distribution=distribution, snr_range=wham_snr_range)
    gain_sampler_saturation = get_gain_sampler(distribution=distribution, gain_range=saturation_gain_range)

    log_noisy_sample(
                log_file=os.path.join(output_root, "dataset_info.json"),
                clean_data=None,
                noise_path=None,
                snr_db=None,
                gain=None,
                distribution=distribution,
                wind_snr_range=wind_snr_range,
                wham_snr_range=wham_snr_range,
                saturation_gain_range=saturation_gain_range,
                audio_ext=audio_ext, 
            )

    # Torch-based transforms
    transforms = [
        Resample(sample_rate),
        Normalize()
    ]

    # Collect noise files
    noise_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(noise_root)
    for f in files
    if f.endswith(".wav")
    ]
    assert len(noise_files) > 0, "No noise files found!"

    for root, _, files in os.walk(speech_root):
        c = 0
        for fname in files:
            if not fname.endswith(audio_ext):
                continue
            c += 1
            clean_path = os.path.join(root, fname)

            # Recreate directory structure
            rel_path = os.path.relpath(root, speech_root)
            out_dir = os.path.join(output_root, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)

            # Load clean speech (torch.Tensor)
            # clean, sr = load_mp3_audio(clean_path, transforms=transforms)
            clean, sr = load_audio(clean_path, transforms)
            assert sr == sample_rate, "must have same sampling rate"

            # Sample SNR
            noise_type = get_noise_type_from_root(root, ENVIRONMENTS)

            if noise_type in targeted_domain: # apply noise only on a targeted domain
                if noise_type in ["wham", "wind"]: # additive noise
                    if noise_type == "wham":
                        snr_db = snr_sampler_wham()
                        gain = None
                        # Pick random noise from WHAM!
                        noise_path = rd.choice(noise_files)
                        noise, sr_noise = load_audio(noise_path, transforms)     
                        assert sr_noise == sample_rate, "must have same sampling rate"
                    elif noise_type == "wind":
                        noise_path = None 
                        snr_db = snr_sampler_wind()
                        gain = None
                        noise = get_simulated_wind()

                    # Mix (torch)
                    mixed = mix_at_snr(clean, noise, snr_db)

                    # Prevent clipping (torch-safe)
                    peak = mixed.abs().max().clamp(min=1.0)
                    mixed = mixed / peak
                
                elif noise_type == "saturation":
                    noise_path = None 
                    snr_db = None
                    gain = gain_sampler_saturation()
                    

                    # Applay saturation
                    mixed = add_saturation(clean, gain)

                # Save (convert to numpy)
                sf.write(
                    out_path,
                    mixed.cpu().numpy(),
                    sr,
                    format="MP3"
                )
                log_noisy_sample(
                    log_file=os.path.join(output_root, "dataset_info.json"),
                    clean_data=fname,
                    noise_path=noise_path,
                    snr_db=snr_db,
                    gain=gain,
                    distribution=None,
                    wind_snr_range=None,
                    wham_snr_range=None,
                    saturation_gain_range=None,
                    audio_ext=None,
                )
            else: # save clean audio for 
                sf.write(
                    out_path,
                    clean.cpu().numpy(),
                    sr,
                    format="MP3"
                )

                log_noisy_sample(
                    log_file=os.path.join(output_root, "dataset_info.json"),
                    clean_data=fname,
                    noise_path=None,
                    snr_db=None,
                    gain=None,
                    distribution=None,
                    wind_snr_range=None,
                    wham_snr_range=None,
                    saturation_gain_range=None,
                    audio_ext=None,
                )

        if c == 0: 
            print(f"folder {root.split('/')[-1]}")
        else:
            print(f"{c} audio have been processed in folder {root.split('/')[-1]}")

def get_data_from_CV(data_path, split):
    all_data = {}
    for l in LANGUAGES: 
        train_path = data_path / l / f"{split}.tsv" 
        df_train = pd.read_csv(train_path, sep="\t", low_memory=False)
        # display(df_train.head(5))
        df_sel = df_train[["client_id", "path", "locale"]].rename(
        columns={
            "client_id": "spk_id",
            "path": "path_to_mp3",
            "locale": "language",
            }
        ).drop_duplicates(subset="spk_id", keep="first")

        # Read clips duration
        clip_duration_path = data_path / l / "clip_durations.tsv"
        df_dur = pd.read_csv(clip_duration_path, sep="\t", header=None, names=["clip", "duration_ms"], low_memory=False)
        df_dur["duration_ms"] = pd.to_numeric(df_dur["duration_ms"], errors="coerce")
        clip_duration = dict(zip(df_dur["clip"], df_dur["duration_ms"]))

        records = df_sel.to_dict(orient="records")
        for rec in records:
            clip_id = Path(rec["path_to_mp3"]).name 
            if clip_duration.get(clip_id) is not None:
                rec["duration_s_before_crop"] = float(clip_duration[clip_id]) / 1000.0
            else:
                rec["duration_s_before_crop"] = None  

        all_data[l] = records
    return all_data

def split_domains(
    environments,
    path_to_CV,
    dataset_size,
    min_duration=3,
    seed=42
):
    rd.seed(seed)
    n_domains = len(environments)

    domains = {env: [] for env in environments}

    data_map = {
        "train": get_data_from_CV(data_path=path_to_CV, split='train'),
        "test": get_data_from_CV(data_path=path_to_CV, split='test'),
        "dev": get_data_from_CV(data_path=path_to_CV, split='dev')
    }

    # Filter all splits
    filtered = {
        split: {
            l: [a for a in audios if a["duration_s_before_crop"] >= min_duration]
            for l, audios in split_data.items()
        }
        for split, split_data in data_map.items()
    }

    # Train proportions
    train_data = filtered["train"]
    total_train = sum(len(v) for v in train_data.values())

    if total_train == 0:
        raise ValueError("No valid training data")

    languages = list(train_data.keys())

    prop_languages = {
        l: len(train_data[l]) / total_train
        for l in languages
    }

    print("\nLanguage proportions (from train):")
    for l, p in prop_languages.items():
        print(f"  {l}: {round(p*100, 2)}%")

    # Sampling with detailed logging
    global_stats = defaultdict(int)

    for l in languages:
        target = int(prop_languages[l] * dataset_size)
        collected = []

        print(f"\nLanguage: {l}")
        print(f"  Target samples: {target}")

        for split in ["train", "dev", "test"]:
            available = filtered.get(split, {}).get(l, [])
            remaining = target - len(collected)

            if remaining <= 0:
                break

            take = min(remaining, len(available))
            if take > 0:
                samples = rd.sample(available, take)
                collected.extend(samples)

                global_stats[split] += take
                print(f"    {split}: {take}/{len(available)}")

        print(f"  → Collected: {len(collected)}/{target}")

        # Split across environments
        for i, env in enumerate(environments):
            domains[env].extend(collected[i::n_domains])

    # Global summary
    print("\n========== GLOBAL SUMMARY ==========")
    total = sum(len(v) for v in domains.values())
    print(f"Total samples: {total}")

    for split in ["train", "dev", "test"]:
        print(f"  {split}: {global_stats[split]}")

    for env in environments:
        print(f"  {env}: {len(domains[env])}")

    return domains

def create_domain_folder(domains, path, path_to_CV, dataset_name): 
    #print(Path(path/dataset_name))
    path_to_dataset = Path(path/dataset_name)
    path_to_dataset.mkdir(parents=True, exist_ok=True)
    json_path = path_to_dataset / "used_files.json"
    with open(json_path, "w") as f:
        json.dump(domains, f, indent=4)

    for d in domains.keys():
        path_to_domain = Path(path/dataset_name/d)
        for a in tqdm(domains[d]):
            path_to_mp3 = path_to_CV / a['language'] / "clips" / a['path_to_mp3']
            index = LANGUAGES.index(a['language'])
            path_to_target = path_to_domain / str(index)       
            path_to_target.mkdir(parents=True, exist_ok=True)
            target = path_to_target / a['path_to_mp3'].split("/")[1]
            if not target.exists():
                shutil.copy2(path_to_mp3, path_to_target)


if __name__ == "__main__":

    ENVIRONMENTS = ["clean", "wham", "wind", "saturation"]
    LANGUAGES = ["fr", "de", "es", "zh-CN"] 
    DATASET_SIZE = 4500

    parser = argparse.ArgumentParser(description="Generate a noisy speech dataset from clean audio and noise files")

    clean_dataset_name = "AUDIO_DA_CWWS_mono"
    noisy_dataset_name = f"NOISY_{clean_dataset_name}"

    

    ## To be removed ! 
    path_to_data = data_directory= Path(os.environ["SCRATCH"]) / "domainbed/data"
    cv_root = Path(path_to_data /"URGENT/corpus/CommonVoice/cv-corpus-22.0-2025-06-20")

    speech_root = Path(path_to_data / f"{clean_dataset_name}")
    wham_noise_root = Path(path_to_data / "URGENT/noises/wham_noise/")   
    output_root = Path(path_to_data / f"{noisy_dataset_name}")
    
    parser.add_argument('--data_dir', type=str, default=path_to_data, help="Root directory to main data directory")
    parser.add_argument("--commonvoice_root", type=str, default=cv_root, help="Root directory containing clean speech (tree preserved)")
    parser.add_argument("--speech_root", type=str, default=speech_root, help="Root directory containing clean speech (tree preserved)")
    parser.add_argument("--noise_root",type=str, default=wham_noise_root, help="Directory containing WHAM! noise .wav files")
    parser.add_argument("--output_root", type=str, default=output_root, help="Output directory where noisy audio will be written")
    ## 
    # parser.add_argument("--commonvoice_root", type=str, required=True, help="Root directory containing .mp3 clean speech from CommonVoice")
    # parser.add_argument("--speech_root", type=str, required=True, help="Root directory containing clean speech (tree preserved)")
    # parser.add_argument("--noise_root",type=str, required=True, help="Directory containing noise wav files")
    # parser.add_argument("--output_root", type=str, required=True, help="Output directory where noisy audio will be written")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--audio_ext", type=str, default=".mp3", help="Audio extension to process/save (default: .mp3)")
    
    parser.add_argument("--snr_distribution", type=str, default="triangular", choices=["uniform", "triangular"], help="SNR distribution")
    parser.add_argument("--targeted_domain", type=str, default=["wham", "wind","saturation"], help="Select on which domain you apply additive noise")

    parser.add_argument("--wind_snr_range", type=str, default=[-40, -30, -20], help="SNR range for sampler - wind noise") 
    parser.add_argument("--wham_snr_range", type=str, default=[-15, -5, 5], help="SNR range for sampler - wind noise") 
    parser.add_argument("--saturation_gain_range", type=str, default=[0.5, 2.5, 4.5], help="gain range for sampler - saturation noise") 

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_root, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_root, 'err.txt'))

    print("=== Configuration ===")
    print(f"Clean speech root: {str(args.speech_root).split('/')[-1]}")
    print(f"Noise files root: {str(args.noise_root).split('/')[-1]}")
    print(f"Output root: {str(args.output_root).split('/')[-1]}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Audio extension: {args.audio_ext}")
    print(f"Wind SNR range: {args.wind_snr_range}")
    print(f"Wham SNR range: {args.wham_snr_range}")
    print(f"Saturation Gain range: {args.saturation_gain_range}")
    print(f"SNR distribution: {args.snr_distribution}")
    print("=====================\n")

    # Ensure output directory exists
    os.makedirs(args.output_root, exist_ok=True)
    print("Output directory ready.\n")

    # Generate domains with clean speech 
    

    # generate clean speech dataset with domain split
    print("Creation of domains with clean speech ...")
    domains = split_domains(ENVIRONMENTS, path_to_CV=args.commonvoice_root, dataset_size=DATASET_SIZE)
    print(f"Copying selected files in {clean_dataset_name} dataset ...")
    create_domain_folder(domains=domains, path=args.data_dir , path_to_CV=args.commonvoice_root, dataset_name=clean_dataset_name)

    # Generate noisy dataset
    print("Starting noisy dataset generation...")
    generate_noisy_dataset(
        targeted_domain=args.targeted_domain,
        speech_root=args.speech_root,
        noise_root=args.noise_root,
        output_root=args.output_root,
        distribution=args.snr_distribution,
        wind_snr_range=args.wind_snr_range,
        wham_snr_range=args.wham_snr_range,
        saturation_gain_range=args.saturation_gain_range,
        sample_rate=args.sample_rate,
        audio_ext=args.audio_ext
    )
    print("\nNoisy dataset generation complete!")
