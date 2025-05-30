import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile

print("numpy version:", np.__version__)
print("scipy version:", scipy.__version__)

samplerate, data = wavfile.read('results\\air\\Fast_Slime_fast_2025-03-07_21.35.05.processed_segment1.wav')
print(f'Sample rate: {samplerate} Hz')
print(f'Data shape: {data.shape}')

if len(data.shape) > 1:
    data = data[:, 0]

times = np.arange(len(data)) / samplerate 

plt.figure(figsize=(12, 4))
plt.plot(times, data)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title('Sygnał audio w dziedzinie czasu')
plt.show()
