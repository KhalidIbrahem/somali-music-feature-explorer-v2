import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_pitch(npz_path):
    d = np.load(npz_path)
    times, f0, f0_interp = d["times"], d["f0"], d["f0_interp"]
    plt.figure(figsize=(12,4))
    plt.plot(times, f0, label="F0 (raw)", linewidth=1)
    plt.plot(times, f0_interp, label="F0 (interp)", linewidth=1, alpha=0.8)
    plt.xlabel("Time (s)"); plt.ylabel("Pitch (Hz)"); plt.legend()
    plt.title("Pitch (F0)")
    plt.tight_layout(); plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize saved feature arrays (.npz).")
    parser.add_argument("--pitch_npz", help="Path to pitch_features.npz to plot pitch.")
    args = parser.parse_args()

    if args.pitch_npz:
        plot_pitch(args.pitch_npz)

if __name__ == "__main__":
    main()
