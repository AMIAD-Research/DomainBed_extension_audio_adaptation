import torch
import os 
import json
import sys 
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import random as rd 
from domainbed.lib import misc 
from domainbed.utils_fr import save_accuracy_per_class
from domainbed.scripts_audio.audio_classes import Resample, Normalize
from domainbed.scripts_audio.preprocess_mixtures import load_audio, mix_at_snr
from domainbed.scripts_audio.sc_wind_noise_generator import WindNoiseGenerator as wng


def save_few_examples(x, snr, noise_type, output_dir): 

    for i in range(min(4, x.size(0))):
        fname = f"{noise_type}_SNR_{snr}_s{i}.mp3"
        out_path = os.path.join(output_dir, fname)
        mixed = x[i,:].squeeze(0)
        sr = 16000 

        sf.write(
            out_path,
            mixed.cpu().numpy(),
            sr,
            format="MP3"
        )
    

def add_wham_noise(x, snr):
    noise_root = Path(os.environ["SCRATCH"]) / "domainbed/data" / "URGENT/noises/wham_noise/"    

    noise_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(noise_root)
    for f in files
    if f.endswith(".wav")
    ]

    transforms = [
        Resample(16000),
        Normalize()
    ]
    noise_path = rd.choice(noise_files)
    noise, sr_noise = load_audio(noise_path, transforms=transforms)

    T = x.shape[-1]
    noise = noise[:T].view(1, 1, -1)
    noise = noise.repeat(x.size(0), 1, 1)  # [64, 1, 48000]

    mixed = mix_at_snr(x, noise, snr)
    peak = mixed.abs().max().clamp(min=1.0)
    mixed = mixed / peak
    
    return mixed

def add_wind_noise(x, snr): 

    batch_size = x.size(0)
    wn_signals = []

    for _ in range(batch_size):
        # 1. Random gustiness
        gustiness_range = [10, 16]  # highly turbulent
        gustiness = np.random.uniform(gustiness_range[0], gustiness_range[1])
        number_points_wind_profile = int(1.5 * gustiness)

        # 2. Random wind profile points
        wind_profile_magnitude_range = [10, 18]
        wind_profile_acceptable_transition_threshold = 2.5

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

        wn_signals.append(wn_signal_tensor)

    # Stack along batch dimension: shape [batch, 1, time]
    wn_signals = torch.cat(wn_signals, dim=0)

    mixed = mix_at_snr(x, wn_signals, snr)
    peak = mixed.abs().max().clamp(min=1.0)
    mixed = mixed / peak

    return mixed

def add_saturation(x, gain):
    """
    x: [B,1,T]
    gain: facteur d'amplification avant saturation
    """

    rms = torch.sqrt(torch.mean(x**2, dim=2, keepdim=True))
    x_norm = x / (rms + 1e-8)

    x_amp = gain * x_norm

    # saturation dure
    sat_x = torch.clamp(x_amp, -1.0, 1.0)

    return sat_x


def modified_accuracy(network, loader, weights, device, name, noise_type, snr, output_dir):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader: 
            if "env1" in name and noise_type == "wham":
                x = add_wham_noise(x,snr)
                save_few_examples(x, snr, noise_type, output_dir)    
            if "env2" in name and noise_type == "wind":
                x = add_wind_noise(x, snr)
                save_few_examples(x, snr, noise_type, output_dir)
            if "env2" in name and noise_type == "saturation":
                x = add_saturation(x, snr)
                save_few_examples(x, snr, noise_type, output_dir)
            x = x.to(device) # [B, 1, 48000]
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

from collections import defaultdict
def modified_accuracy_per_class(network, loader, weights, device, name, noise_type, snr):
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            if "env1" in name and noise_type == "wham":
                x = add_wham_noise(x,snr)
            if "env2" in name and noise_type == "wind":
                x = add_wind_noise(x, snr)
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x), device=device)
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)].to(device)
                weights_offset += len(x)
            preds = p.argmax(1)
            for cls in y.unique():
                cls = cls.item()
                mask = (y == cls)
                correct_per_class[cls] += ((preds[mask] == y[mask]).float() * batch_weights[mask]).sum().item()
                total_per_class[cls] += batch_weights[mask].sum().item()
    network.train()
    # Compute accuracy per class
    acc_per_class = {
        cls: correct_per_class[cls] / total_per_class[cls]
        for cls in total_per_class
        if total_per_class[cls] > 0
    }

    return acc_per_class


def noise_inference(algorithm, evaluate, device, noise_type):

    eval_loader_names, eval_loaders, eval_weights = evaluate 

    if noise_type == "wham":
        path_to_ckpt = Path(os.environ["SCRATCH"]) / "domainbed/results/noisy_wham_train_audio_classif/ref_clean_test_env_1"
    elif noise_type == "wind":
        path_to_ckpt = Path(os.environ["SCRATCH"]) / "domainbed/results/noisy_wind_train_audio_classif/ref_clean_test_env_2"
    else: 
        raise ValueError(f"Unknown reference for this noise type")
    assert os.path.isdir(path_to_ckpt), "Checkpoint directory not found"
    path_to_pkl = os.path.join(path_to_ckpt, "model.pkl")
    assert os.path.isfile(path_to_pkl), "model.pkl not found in checkpoint directory"

    # Load the algorithm with the corresponding checkpoint 
    ckpt = torch.load(path_to_pkl, map_location=device)
    algorithm.load_state_dict(ckpt["model_dict"])
    algorithm.to(device)

    # Modify the evals loader with corresponding SNR
    SNRs_wham = [-20, -15, -10, -7.5, -5, -2.5, 0, 5, 10, 20]
    SNRs_wind = [-50, -40, -30, -25, -20, -15, -10, 0, 10]
    gain_saturation = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    if noise_type == "wham":
        SNRs = SNRs_wham
        output_dir = f"{path_to_ckpt}/{noise_type}_snr_vs_perf"
    elif noise_type == "wind":
        SNRs = SNRs_wind
        output_dir = f"{path_to_ckpt}/{noise_type}_snr_vs_perf"
    elif noise_type == "saturation":
        SNRs = gain_saturation
        output_dir = f"{path_to_ckpt}/{noise_type}_snr_vs_perf"
    else: 
        raise ValueError(f"Unknown reference for this noise type")


    # Infer with the algorithm over evals loader 
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

    last_results_keys = None
    for snr in SNRs:
        results = {
            "SNR": snr,
        }

        results_per_class = {
            "SNR": snr,
        }

        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            acc = modified_accuracy(algorithm, loader, weights, device, name, noise_type, snr, output_dir)
            results[name+'_acc'] = acc
            acc_per_class = modified_accuracy_per_class(algorithm, loader, weights, device, name, noise_type, snr)
            for cls, v in acc_per_class.items():
                results_per_class[f"{name}/acc_class_{cls}"] = v

        results_keys = sorted(results.keys())
        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        save_accuracy_per_class(os.path.join(output_dir, 'out_per_class.txt'), results_per_class)
        
    with open(os.path.join(output_dir, 'verif_done'), 'w') as f:
        f.write('done')
    return exit(0)