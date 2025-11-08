# Somali Music Feature Explorer
#
A starter project to extract and visualize audio features from Somali music recordings.
It produces waveform, tempo & beats, pitch (PYIN), mel‑spectrogram, and MFCCs, and saves plots and feature arrays.

## Structure
```
somali-music-feature-explorer/
├── data/            # put short audio clips (e.g., ≤30s for fair use/testing)
├── features/        # saved feature arrays (.npz)
├── reports/         # saved plots (.png)
├── notebooks/
│   └── 01_feature_explorer.ipynb
├── src/
│   ├── extract_features.py
│   └── visualize.py
└── requirements.txt
```
## Quickstart
1) Create a virtual env (recommended), then install:
```
pip install -r requirements.txt
```
2) Put a WAV/MP3 file into `data/` (e.g., `data/song.wav`).  
3) Run the notebook `notebooks/01_feature_explorer.ipynb` OR run the CLI:

```
python src/extract_features.py --audio data/song.wav
```

Outputs:
- Plots in `reports/`: `waveform_with_beats.png`, `pitch_f0.png`, `mel_spectrogram.png`, `mfcc.png`
- Feature arrays in `features/`: `pitch_features.npz`, `mel_spectrogram.npz`, `mfcc_features.npz`, `beat_sync_mfcc.npz`

## Notes
- This project uses **librosa** and **matplotlib** only (no seaborn).
- Keep audio clips short for quick iteration and fair‑use testing.
- Extend this repo with classification or generative models next.
# somali-music-feature-explorer-
