import torch
import torchaudio
import random
from pathlib import Path
import soundfile as sf
import os

class AudioCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform, sample_rate):
        # output_folder = "/lustre/fsn1/projects/rech/ldz/upr55xw/domainbed/listen_audio"
        # os.makedirs(output_folder, exist_ok=True)
        # output_path = os.path.join(output_folder, "before_transformations.mp3")
        # torchaudio.save(output_path, waveform, sample_rate, format="mp3", bitrate=192)

        for t in self.transforms:
            waveform = t(waveform, sample_rate)
        
        # output_path = os.path.join(output_folder, "after_transformations.mp3")
        # torchaudio.save(output_path, waveform, sample_rate, format="mp3", bitrate=192)
        return waveform

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

    
class RandomTimeShift:
    def __init__(self, max_shift_pct=0.1):
        self.max_shift_pct = max_shift_pct

    def __call__(self, data):
        waveform, sr = data
        max_shift = int(self.max_shift_pct * waveform.shape[-1])
        shift = random.randint(-max_shift, max_shift)
        waveform = torch.roll(waveform, shifts=shift, dims=-1)
        return waveform, sr

class RandomGain:
    def __init__(self, min_gain=0.8, max_gain=1.2):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, data):
        waveform, sr = data
        gain = random.uniform(self.min_gain, self.max_gain)
        waveform = waveform * gain
        return waveform, sr


class AddNoise:
    def __init__(self, noise_level=0.005):
        self.noise_level = noise_level

    def __call__(self, data):
        waveform, sr = data
        noise = torch.randn_like(waveform) * self.noise_level
        waveform = waveform + noise
        return waveform, sr


class RandomCropPad:
    def __init__(self, target_len):
        self.target_len = target_len

    def __call__(self, data):
        waveform, sr = data
        length = waveform.shape[-1]

        if length > self.target_len:
            start = random.randint(0, length - self.target_len)
            waveform = waveform[..., start:start + self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform, sr

class DropSampleRate:
    def __call__(self, data):
        waveform, _ = data
        return waveform