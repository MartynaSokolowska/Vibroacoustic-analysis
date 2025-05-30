import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile


def audio_plot(data, samplerate):
    if len(data.shape) > 1:
        data = data[:, 0]

    times = np.arange(len(data)) / samplerate 

    plt.figure(figsize=(12, 4))
    plt.plot(times, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Audio signal in the time domain')
    plt.show()


def plot_umap(X_2d, labels):
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label)
    plt.legend()
    plt.title("UMAP Projection")
    plt.show()
