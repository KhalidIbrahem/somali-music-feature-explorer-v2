import argparse
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def ensure_dirs(features_dir, reports_dir):
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

def extract_all(audio_path, features_dir="features", reports_dir="reports", sr=22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y)/sr
    ensure_dirs(features_dir, reports_dir)

    # Tempo & beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Waveform with beats
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y, sr=sr)
    for bt in beat_times:
        plt.axvline(bt, linestyle='--', alpha=0.4)
    plt.title(f"Waveform with Beat Grid (Tempo ≈ {tempo:.1f} BPM)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "waveform_with_beats.png"), dpi=150)
    plt.close()

    # Pitch with PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    f0_interp = f0.copy()
    idx = np.where(~np.isnan(f0))[0]
    if len(idx) > 1:
        f0_interp = np.interp(np.arange(len(f0)), idx, f0[idx])

    plt.figure(figsize=(12,4))
    plt.plot(times, f0, label="F0 (raw)", linewidth=1)
    plt.plot(times, f0_interp, label="F0 (interp)", linewidth=1, alpha=0.8)
    plt.xlabel("Time (s)"); plt.ylabel("Pitch (Hz)"); plt.legend()
    plt.title("Pitch (F0) Estimation (PYIN)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "pitch_f0.png"), dpi=150)
    plt.close()

    np.savez(os.path.join(features_dir, "pitch_features.npz"),
             times=times, f0=f0, f0_interp=f0_interp,
             voiced_flag=voiced_flag, voiced_probs=voiced_probs, sr=sr)

    # Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=128, fmin=20, fmax=sr//2)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.title("Mel-Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "mel_spectrogram.png"), dpi=150)
    plt.close()

    np.savez(os.path.join(features_dir, "mel_spectrogram.npz"), S=S, S_db=S_db, sr=sr)

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    plt.figure(figsize=(12,4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.title("MFCC (13 Coefficients)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "mfcc.png"), dpi=150)
    plt.close()

    np.savez(os.path.join(features_dir, "mfcc_features.npz"),
             mfcc=mfcc, mfcc_delta=mfcc_delta, mfcc_delta2=mfcc_delta2, sr=sr)

    # Beat-synchronous MFCC means
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)
    if len(beat_times) < 2:
        beat_times = np.arange(0, duration, 1.0)
    beat_boundaries = np.append(beat_times, duration)
    beat_mfcc_means = []
    for i in range(len(beat_boundaries)-1):
        start_t, end_t = beat_boundaries[i], beat_boundaries[i+1]
        idx = np.where((frame_times >= start_t) & (frame_times < end_t))[0]
        if len(idx):
            beat_mfcc_means.append(mfcc[:, idx].mean(axis=1))
        else:
            beat_mfcc_means.append(np.full((mfcc.shape[0],), np.nan))
    beat_mfcc_means = np.array(beat_mfcc_means)
    np.savez(os.path.join(features_dir, "beat_sync_mfcc.npz"),
             beat_times=beat_times, beat_mfcc_means=beat_mfcc_means)

    return {"tempo": float(tempo), "beats": len(beat_times), "duration_sec": float(duration)}

def main():
    parser = argparse.ArgumentParser(description="Extract audio features and save plots/arrays.")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav or mp3).")
    parser.add_argument("--features_dir", default="features")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    stats = extract_all(args.audio, args.features_dir, args.reports_dir, args.sr)
    print(f"Done. Stats: {stats}")

if __name__ == "__main__":
    main()
