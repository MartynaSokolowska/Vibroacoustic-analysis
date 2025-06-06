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
        movement_type: str, 
        layout_type: str, 
        input_file: str, 
        output_dir: str, 
        preprocess_func=lambda x, _: x, 
        segment_duration_ms = 100
        ):
    basename = Path(input_file).stem
    sr, audio = wavfile.read(input_file)
    audio = preprocess_func(audio, sr)
    try:
        segments = segmentations[movement_type][layout_type]
    except KeyError:
        raise ValueError("Invalid movement_type or layout_type")

    for material in segments:
        for start, end in (material.segments):
            start_ms = int(start * sr)
            end_ms = int(end * sr)
            segment_audio = audio[start_ms:end_ms]

            chunk_size = int(sr * segment_duration_ms / 1000)
            num_chunks = (len(segment_audio) // chunk_size)

            chunks = np.array([
                segment_audio[j * chunk_size:(j + 1) * chunk_size]
                for j in range(num_chunks)
            ])

            rms_values = np.array([
                np.sqrt(np.mean(chunk.astype(np.float64)**2)) for chunk in chunks
            ])

            mean_rms = np.mean(rms_values)
            std_rms = np.std(rms_values)

            valid_indices = np.where(
                (rms_values >= mean_rms - 2 * std_rms) &
                (rms_values <= mean_rms + 2 * std_rms)
            )[0]

            folder_name = os.path.join(
                output_dir,
                material.material_name.replace('/', '_').replace(' ', '_')
            )
            os.makedirs(folder_name, exist_ok=True)

            for j, idx in enumerate(valid_indices, start=1):
                chunk = chunks[idx]
                output_path = os.path.join(folder_name, f"{basename}_chunk{j}.wav")
                wavfile.write(output_path, sr, chunk.astype(np.int16))
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