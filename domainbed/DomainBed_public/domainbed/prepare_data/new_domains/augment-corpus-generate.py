"""
Generation of an augmented version of a speech corpus : audio generation.
"""

import argparse
import numpy as np
import os
import pandas as pd


import random as rd 
import os
import soundfile as sf
import torch
from pathlib import Path
import json 
import torchaudio




import scipy as sp



import multiprocessing as mp

from spectrum.linear_prediction import lsf2poly

# audio processing
class Resample:
    def __init__(self, target_sr):
        self.target_sr = target_sr

    def __call__(self, data):
        waveform, sr = data

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sr
            )

        return waveform, self.target_sr

class Normalize:
    def __call__(self, data):
        waveform, sr = data
        max_val = waveform.abs().max().clamp(min=1e-6)
        waveform = waveform / max_val

        return waveform, sr
# wind generator





# def lsf2poly(lsf):
#     # Convert LSFs to complex roots on unit circle
#     roots = np.exp(1j * lsf)

#     # Ensure conjugate symmetry (real polynomial)
#     roots = np.concatenate([roots, np.conj(roots)])

#     # Convert roots to polynomial coefficients
#     a = np.poly(roots)

#     # Keep real part (numerical cleanup)
#     return np.real(a)

class WindNoiseGenerator:
    """Wind Noise Generator Class"""

    def __init__(self, fs=48000, duration=5, generate=True, wind_profile=None, gustiness=3, short_term_var=True, start_seed=None):
        """Initizalize object"""

        self.fs = fs
        self.duration = duration
        self.samples = fs * duration
        self.generate = generate
        self.gustiness = gustiness
        self.wind_profile = wind_profile
        self.short_term_var = short_term_var
        if start_seed is not None:
            np.random.seed(start_seed)

    def generate_wind_noise(self):
        """Generate single-channel wind noise by filtering excitation signal"""

        if self.generate:
            wind_profile = self._generate_wind_speed_profile()
        else:
            wind_profile = self._import_wind_speed_profile()

        exc = self.generate_excitation_signal(wind_profile)
        exc_filtered = self._filter(exc, wind_profile, 2048)
        exc_filtered = 0.95*exc_filtered / \
            max(np.abs(exc_filtered))

        return exc_filtered, wind_profile

    def generate_excitation_signal(self, wind_profile):
        """Generate excitation signal"""

        window_size = 128
        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        wgn = np.concatenate(
            (np.zeros(window_size), np.random.randn(self.samples), np.zeros(window_size)))
        wgn_length = len(wgn)

        lt_var = self._generate_long_term_variance(wind_profile)
        lt_var = np.concatenate((np.zeros(window_size), lt_var, np.zeros(window_size)))

        st_var = self._generate_short_term_variance_garch(wind_profile)
        cond_var = np.abs(st_var)

        num_windows = (wgn_length - window_size) // hops + 1
        exc = np.zeros(wgn_length)

        for time_frame in range(num_windows-1):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            gain_ltst = lt_var[idx]
            if self.short_term_var:
                gain_ltst *= np.sqrt(cond_var[time_frame])
            noise_seg_ltst = gain_ltst * wgn[idx] * hann_window
            exc[idx] += noise_seg_ltst

        exc = exc[window_size:-window_size]

        return exc

    def _generate_short_term_variance_garch(self, wind_profile):
        """Generate short-term variance of GARCH process"""

        window_size = 128
        hops = window_size // 2 # overlap

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))
        profile_length = len(profile)

        num_windows = (profile_length - window_size) // hops + 1
        st_var = np.zeros(num_windows)
        cond_var = np.zeros(num_windows)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            alpha, beta, omega = self._speed2par(speed)

            if alpha + beta > 1:
                beta = 0

            cond_var[time_frame] = omega + alpha * \
                st_var[time_frame-1]**2 + beta*(cond_var[time_frame-1])
            st_var[time_frame] = np.sqrt(np.abs(cond_var[time_frame])) * \
                np.random.randn()

        return st_var/max(np.abs(st_var))

    def _generate_long_term_variance(self, wind_profile):
        """Generate long-term variance"""

        # Regression parameter noise variance/wind speed
        regression_coeff = np.array([8.00071114414022, -220.332082908370])

        # Long-term noise variance based on wind speed profile in dB scale
        variance_profile_db = np.polyval(regression_coeff, wind_profile)

        # Long-term noise variance in linear scale
        variance_profile = 10 ** (variance_profile_db / 10)
        var_lt = np.sqrt(np.abs(variance_profile))  # long-term gain

        return var_lt

    def _generate_wind_speed_profile(self, b_par=2, a_par=2):
        """Generate the wind speed profile by sampling a Weibull distribution"""

        speed_points = int(
            self.gustiness)  # gustiness, 1 = constant speed, 10 = highly-variable speed

        # Sample from the Weibull distribution (change b and a for different distributions)
        wind_speed_profile_lt = b_par * np.random.weibull(a_par, speed_points)

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)

        # Additive speed fluctuations
        fluctuations = 10 * np.random.randn(self.samples)

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _import_wind_speed_profile(self):
        """Read the wind speed profile from input"""

        wind_speed_profile_lt = self.wind_profile  # load speed values

        # Interpolate speed values as required audio samples
        wind_speed_profile = sp.signal.resample(
            wind_speed_profile_lt, self.samples)
        fluctuations = 10 * np.random.randn(self.samples)  # additive speed fluctuations

        # Smoothing of the fluctuations
        hann_window = np.hanning(self.fs * 100e-3)
        hann_window /= sum(hann_window)  # hanning window for the smoothing
        fluctuations = sp.signal.lfilter(hann_window, 1, fluctuations)

        # Add the fluctuations to the generated wind speed profile
        wind_speed_profile += fluctuations

        return wind_speed_profile

    def _filter(self, exc, wind_profile, window_size):
        """Filter the excitation signals with the AR filter coefficients"""

        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))

        exc = np.concatenate((np.zeros(window_size), exc, np.zeros(window_size)))
        exc_length = len(exc)

        # Overlap-add approach for the time-varying filtering of the excitation signal
        num_windows = (exc_length - window_size) // hops + 1
        exc_filtered = np.zeros(exc_length)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            lpc = self._lsf2lpc(speed)

            exc_seg = exc[idx] * hann_window
            exc_seg_filtered = sp.signal.lfilter(
                np.array([1.0]), lpc, exc_seg)

            exc_filtered[idx] += exc_seg_filtered

        exc_filtered = exc_filtered[window_size:-window_size]

        return exc_filtered

    def _speed2par(self, speed):
        """Convert speed to GARCH parameters"""

        gp_alpha = np.array([-2.73244444508231e-05, 0.00141129711949206, -
                            0.0274652794467908, 0.257613241095714, -0.139824587447063])
        gp_beta = np.array(
            [-9.75160902595897e-05, 0.00464300106846736, -0.0871968755558256, 0.651013973757802])
        gp_omega = np.array(
            [9.69585296574741e-05, -0.00231853830578967, 0.0124681159197788])

        alpha = np.polyval(gp_alpha, speed)
        beta = np.polyval(gp_beta, speed)
        omega = np.polyval(gp_omega, speed)

        return alpha, beta, omega

    def _lsf2lpc(self, speed):
        """Generate LPC coefficients from the LSF-speed models given a speed value"""

        # Regression coefficients of the LFS-speed model
        # The n-th LFS coefficient corresponds to the n-th column
        regression_coeff = np.array([[-2.63412497797108e-06, 5.93162248595821e-05,
                                      0.000215613938043173, -0.000149723789407121,
                                      -0.000213703084399375],
                                     [9.50240139044154e-05,	-0.00271741166649528,
                                      -0.0103783584000284, 0.00483963669507075,
                                      0.00931864887930701],
                                     [-0.000699199223507821, 0.0428714179385289,
                                      0.177250839818556, -0.0329542145779793,
                                      -0.129910107562929],
                                     [0.0106849674771013, -0.234688122194936,
                                      -1.21337646113093, -0.168053225019258,
                                      0.568371362156217],
                                     [-0.000966851130291645, 0.541693139684727,
                                      3.24796925730457, 2.54984352038733,
                                      1.86097523205089]])
        order = 5

        # Estimate LFS based on the speed value
        lfs_estimated = np.zeros(order)

        for order_idx in range(order):
            lfs_estimated[order_idx] = np.polyval(regression_coeff[:, order_idx], speed)

        # Convert LFS into LPC coefficients
        lpc_a = lsf2poly(lfs_estimated)

        return lpc_a




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
    wn_instance = WindNoiseGenerator(
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

def parse_args():

    """ parse_args """

    parser = argparse.ArgumentParser(
        description='Augmentation')
    
    parser.add_argument(
        "--input_rootdir", 
        type=str,
        help="input rootdir"
    )

    parser.add_argument(
        "--original_rootdir", 
        type=str,
        help="original rootdir"
    )

    parser.add_argument(
            "--n", 
            type=int,
            help="n process"
        )

    return parser.parse_args()

SAMPLE_RATE = 16000

transforms = [
        Resample(SAMPLE_RATE),
        Normalize()
    ]



if __name__=="__main__":
    args = vars(parse_args())
    input_rootdir = args["input_rootdir"]
    original_rootdir = args["original_rootdir"]
    n = args["n"]

    table_annotation = pd.read_csv(f"{input_rootdir}/annotation/annotation.csv", sep=",")

    augmentation = table_annotation["augmentation"].iloc[0]


    def process_saturation(row):
        audio_path = row["audio_path"]
        gain = row["gain"]
        
        clean_path = original_rootdir + "/" + audio_path
        out_path = input_rootdir + "/" + audio_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clean, sr = load_audio(clean_path, transforms)
        mixed = add_saturation(clean, gain)
        sf.write(
                out_path,
                mixed.cpu().numpy(),
                sr,
                format="flac"
                )

    def process_wind(row):
        audio_path = row["audio_path"]
        snr_db = row["snr_db"]
        noise = get_simulated_wind()
     
        clean_path = original_rootdir + "/" + audio_path
        out_path = input_rootdir + "/" + audio_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clean, sr = load_audio(clean_path, transforms)
        mixed = mix_at_snr(clean, noise, snr_db)
        sf.write(
                out_path,
                mixed.cpu().numpy(),
                sr,
                format="flac"
                )


        
    def process_noise(row):
        audio_path = row["audio_path"]
        snr_db = row["snr_db"]
        noise_path = row["noise"]
        noise, sr_noise = load_audio(noise_path, transforms)   

        
        clean_path = original_rootdir + "/" + audio_path
        out_path = input_rootdir + "/" + audio_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clean, sr = load_audio(clean_path, transforms)
        mixed = mix_at_snr(clean, noise, snr_db)
        sf.write(
                out_path,
                mixed.cpu().numpy(),
                sr,
                format="flac"
                )

        

    if augmentation == "saturation":
        process_row = process_saturation
    elif augmentation == "wind":
        process_row = process_wind
    elif augmentation == "wham":
        process_row = process_noise
    else:
        print("Unknown augmentation ", augmentation)
    

    rows = table_annotation.to_dict("records")
    with mp.Pool(processes=n) as pool:
        results = pool.map(process_row, rows)
    print("Created augmented corpus ", input_rootdir)

    #