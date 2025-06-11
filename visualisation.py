import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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


def plot2D(X_2d, labels, title="2D Projection"):
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label)
    plt.legend()
    plt.title(title)
    plt.show()


def show_confusion_matrix(y_pred, y_true, display_labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.show()
