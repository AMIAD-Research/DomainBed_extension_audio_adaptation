import numpy as np
import torch
import soundfile as sf
import torchaudio
from domainbed.prepare_data.sc_wind_noise_generator import WindNoiseGenerator as wng

def get_simulated_wind(gustiness, wind_profile):
    # gustiness_range = [8, 11]  # highly turbulent
    # gustiness = np.random.uniform(gustiness_range[0], gustiness_range[1])
    number_points_wind_profile = int(1.5 * gustiness)

    # 2. Random wind profile points
    wind_profile_magnitude_range = [8, 15]
    wind_profile_acceptable_transition_threshold = 1.5

    # wind_profile = [
    #     np.random.uniform(
    #         wind_profile_magnitude_range[0], wind_profile_magnitude_range[1]
    #     )
    # ]

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
    # sat_x = torch.clamp(x_amp, -1.0, 1.0)
    sat_x = torch.tanh(x_amp)

    return sat_x


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



def get_train_speakers(
    train_cells,
):
    """
    Return the global set of speakers already used by
    train_small and train_mono.

    train_small/train_mono overlap is allowed, therefore
    we simply take the UNION of both datasets.

    This set is used to forbid any train speaker from
    appearing in validation or test.
    """

    train_speakers = set()

    for speakers in train_cells.values():
        train_speakers.update(
            str(speaker)
            for speaker in speakers
        )

    return train_speakers


def get_validation_speakers(
    validation_records,
):
    """
    Return the global set of speakers already used by validation.
    """

    validation_speakers = set()

    for records in validation_records.values():

        for record in records:

            validation_speakers.add(
                str(record["spk_id"])
            )

    return validation_speakers