from segments import *
from pathlib import Path
import os
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from noisereduce import reduce_noise
import numpy as np
from enum import Enum


class FilterMode(Enum):
    NONE = 0
    BANDPASS = 1
    DENOISE = 2

# FILTER = FilterMode.NONE
# FILTER = FilterMode.BANDPASS
FILTER = FilterMode.DENOISE

def filter_audio_bandpass(audio, sr, low=11000, high=15000):
    b, a = butter(7, [low, high], btype="bandpass", fs=sr)
    return lfilter(b, a, audio)


def reduce_noise_in_audio(audio, sr):
    return np.array(
        (
            reduce_noise(y=audio[:, 0], sr=sr, prop_decrease=0.75),
            reduce_noise(y=audio[:, 1], sr=sr, prop_decrease=0.75),
        )
    ).T


def split_wav_by_material(
        movement_type: str, layout_type: str, input_file: str, output_dir: str, preprocess_func=lambda x, _: x):
    basename = Path(input_file).stem
    sr, audio = wavfile.read(input_file)
    audio = preprocess_func(audio, sr)
    try:
        segments = segmentations[movement_type][layout_type]
    except KeyError:
        raise ValueError("Invalid movement_type or layout_type")

    for material in segments:
        for i, (start, end) in enumerate(material.segments, start=1):
            start_ms = int(start * sr)
            end_ms = int(end * sr)

            folder_name = os.path.join(output_dir, material.material_name.replace('/', '_').replace(' ', '_'))
            os.makedirs(folder_name, exist_ok=True)

            output_path = os.path.join(folder_name, f"{basename}_segment{i}.wav")
            wavfile.write(output_path, sr, audio[start_ms:end_ms])
            print(f"Saved: {output_path}")


def split_all():
    zucchini_top_path = Path(
        os.path.join("data", "01", "chicken-zuchini-experiments_07-10_March_2025", "ChickenBottom_ZuchiniTop"))
    zucchini_bottom_path = Path(
        os.path.join("data", "01", "chicken-zuchini-experiments_07-10_March_2025", "ChickenTop_ZuchiniBottom"))

    paths = [
        (zucchini_top_path, "zucchini_top"),
        (zucchini_bottom_path, "zucchini_bottom")
    ]

    speed_mapping = {
        "speed_10": "slow",
        "speed_15": "medium",
        "speed_25": "fast"
    }
    for base_path, layout_type in paths:
        for speed_dir in base_path.glob("speed_1*"):
            movement_type = speed_mapping.get(speed_dir.name, "slow")
            for wav_file in speed_dir.glob("*.processed.wav"):
                match FILTER:
                    case FilterMode.NONE:
                        split_wav_by_material(movement_type, layout_type, str(wav_file), "results")
                    case FilterMode.BANDPASS:
                        split_wav_by_material(movement_type, layout_type, str(wav_file), "results_filtered",
                                              preprocess_func=filter_audio_bandpass)
                    case FilterMode.DENOISE:
                        split_wav_by_material(movement_type, layout_type, str(wav_file), "results_denoised",
                                              preprocess_func=reduce_noise_in_audio)


split_all()