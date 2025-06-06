from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import numpy as np
import os
import glob


def extract_log_fft_features(wav_path, fixed_length=50):
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    spectrum = fft(audio)
    mag = np.abs(spectrum)
    log_mag = np.log1p(mag)

    reconstructed = np.real(ifft(log_mag))

    segment_size = 1024
    features = [
        np.mean(reconstructed[i : i + segment_size])
        for i in range(0, len(reconstructed), segment_size)
    ]

    if len(features) > fixed_length:
        features = features[:fixed_length]
    else:
        features = features + [0] * (fixed_length - len(features))

    return np.array(features)


def get_all_features(data_root):
    labels = []
    features = []

    for material in os.listdir(data_root):
        folder = os.path.join(data_root, material)
        if not os.path.isdir(folder):
            continue

        for file in glob.glob(f"{folder}/*.wav"):
            feat = extract_log_fft_features(file)
            features.append(feat)
            labels.append(material)

    return np.array(features), np.array(labels)
