# Vibroacoustic-analysis
This project aims to analyze audio recordings captured from various biological and synthetic materials (e.g., chicken, zucchini, gelatin, air) during vibroacoustic experiments. The goal is to perform unsupervised clustering to distinguish between different materials based on their acoustic properties.

### Prerequisites for Running the Analysis

Before executing the main analysis code, it is necessary to preprocess the audio data by segmenting the WAV files. This segmentation is performed by the script `data_transformation.py`, which contains the function `split_all()` responsible for this task.

Important:

- The segmentation function is designed to work exclusively with datasets containing zucchini and chicken audio samples.

- The input directories provided in split_all() must contain the subfolders: speed_10, speed_15, and speed_25, each holding the corresponding WAV files.

Please ensure that your data is organized accordingly before running the segmentation step to guarantee correct processing in subsequent stages.
