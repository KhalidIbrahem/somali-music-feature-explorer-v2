import argparse
import os
import numpy as np
import librosa
from numpy.linalg import norm
from mido import Message, MidiFile, MidiTrack, MetaMessage

# ---------- Helpers ----------

def hz_to_midi(hz):
    hz = np.asarray(hz, dtype=float)
    return 69 + 12 * np.log2(np.maximum(hz, 1e-12) / 440.0)

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_to_name(m):
    m = int(round(m))
    name = NOTE_NAMES[m % 12]
    octave = m // 12 - 1
    return f"{name}{octave}"

# Chromatic movable-do solfege (ascending semitones)
CHROMATIC_SOLFEGE = {
    0:'do', 1:'di', 2:'re', 3:'ri', 4:'mi',
    5:'fa', 6:'fi', 7:'so', 8:'si', 9:'la',
    10:'li', 11:'ti'
}

# Krumhansl key profiles (major / minor)
KR_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KR_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def rotate(arr, n):
    return np.roll(arr, -n)

def key_from_hist(pc_hist):
    """Return (key_name, mode) choosing best correlation with Krumhansl profiles."""
    best_r = -1e9
    best = ("C", "major", 0)
    for tonic in range(12):
        r_major = np.corrcoef(pc_hist, rotate(KR_MAJOR, tonic))[0,1]
        r_minor = np.corrcoef(pc_hist, rotate(KR_MINOR, tonic))[0,1]
        if r_major > best_r:
            best_r = r_major
            best = (NOTE_NAMES[tonic], "major", tonic)
        if r_minor > best_r:
            best_r = r_minor
            best = (NOTE_NAMES[tonic], "minor", tonic)
    return best  # (tonic_name, mode, tonic_pc)

def solfege_for_pitchclass(pc, tonic_pc):
    """Movable-do chromatic solfege relative to tonic pitch class."""
    degree = (pc - tonic_pc) % 12
    return CHROMATIC_SOLFEGE[int(degree)]

# ---------- Transcription core ----------

def track_to_notes(
    y, sr=22050,
    fmin='C2', fmax='C7',
    frame_hop=256,
    min_note_ms=80,
    voicing_prob_thresh=0.5,
    median_win=5
):
    """
    1) F0 with PYIN
    2) Smooth & quantize to semitone
    3) Segment into note events
    Returns: list of notes dicts with onset, offset, midi, pc (pitch class)
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax)
    )
    times = librosa.times_like(f0, sr=sr)

    # Keep only voiced frames with enough probability
    voiced = (voiced_flag == True) & (np.nan_to_num(voiced_prob) >= voicing_prob_thresh)
    f0_clean = np.where(voiced, f0, np.nan)

    # Convert to MIDI and median-filter smooth (ignore NaNs)
    midi = hz_to_midi(f0_clean)
    # fill NaNs for filtering, then put back
    midi_filled = np.where(np.isnan(midi), np.interp(
        np.flatnonzero(np.isnan(midi)),
        np.flatnonzero(~np.isnan(midi)) if np.any(~np.isnan(midi)) else [0],
        midi[~np.isnan(midi)] if np.any(~np.isnan(midi)) else [69.0]
    ), midi)
    if median_win > 1:
        from scipy.ndimage import median_filter
        midi_smooth = median_filter(midi_filled, size=median_win)
    else:
        midi_smooth = midi_filled
    midi_smooth = np.where(voiced, midi_smooth, np.nan)

    # Quantize to nearest semitone
    midi_round = np.rint(midi_smooth)

    # Segment: group consecutive frames with same semitone
    notes = []
    in_note = False
    start_idx = 0
    cur_pitch = None

    for i in range(len(midi_round)+1):
        same = (i < len(midi_round) and not np.isnan(midi_round[i]) and
                (cur_pitch is None or midi_round[i] == cur_pitch))
        if not in_note:
            if i < len(midi_round) and not np.isnan(midi_round[i]):
                # start a new note
                in_note = True
                start_idx = i
                cur_pitch = midi_round[i]
        else:
            if not same:
                # end current note at i-1
                onset = times[start_idx]
                offset = times[i-1] if i-1 < len(times) else times[-1]
                dur_ms = (offset - onset) * 1000
                if dur_ms >= min_note_ms:
                    midi_note = int(cur_pitch)
                    pc = midi_note % 12
                    notes.append({
                        "onset": float(onset),
                        "offset": float(offset),
                        "midi": int(midi_note),
                        "pc": int(pc)
                    })
                # reset
                in_note = False
                cur_pitch = None
            else:
                # continue same note
                pass

    return notes, times, midi_smooth

def pitchclass_hist(notes):
    hist = np.zeros(12, dtype=float)
    total = 0.0
    for n in notes:
        dur = n["offset"] - n["onset"]
        hist[n["pc"]] += max(dur, 1e-3)
        total += max(dur, 1e-3)
    if total > 0:
        hist /= total
    return hist

def detect_key_from_notes(notes):
    if not notes:
        return ("C","major",0)
    pc_hist = pitchclass_hist(notes)
    return key_from_hist(pc_hist)

def write_midi(notes, out_path, tempo_bpm=90):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    # set tempo
    # default ticks_per_beat=480; convert seconds to ticks
    ticks_per_beat = mid.ticks_per_beat
    track.append(MetaMessage('set_tempo', tempo=librosa.beat.tempo_to_tempo(tempo_bpm)))

    # naive: map seconds to ticks by fixed tempo
    # mido expects delta-times in ticks; we’ll compute relative deltas
    def sec_to_ticks(sec):
        # tempo in microsec per beat:
        us_per_beat = 60_000_000 / tempo_bpm
        ticks_per_sec = ticks_per_beat / (us_per_beat / 1e6)
        return int(round(sec * ticks_per_sec))

    t_prev = 0.0
    for n in notes:
        dt_on = sec_to_ticks(n["onset"] - t_prev)
        track.append(Message('note_on', note=n["midi"], velocity=90, time=dt_on))
        dt_off = sec_to_ticks(n["offset"] - n["onset"])
        track.append(Message('note_off', note=n["midi"], velocity=64, time=dt_off))
        t_prev = n["offset"]

    mid.save(out_path)

def run(audio_path, out_csv, out_midi, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    notes, times, midi_curve = track_to_notes(y, sr=sr)

    # key detection
    tonic_name, mode, tonic_pc = detect_key_from_notes(notes)

    # write CSV
    lines = ["onset_s,offset_s,midi,note,solfege"]
    for n in notes:
        name = midi_to_name(n["midi"])
        sol = solfege_for_pitchclass(n["pc"], tonic_pc)
        lines.append(f"{n['onset']:.3f},{n['offset']:.3f},{n['midi']},{name},{sol}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # write MIDI
    os.makedirs(os.path.dirname(out_midi), exist_ok=True)
    write_midi(notes, out_midi, tempo_bpm=90)

    print(f"Detected key: {tonic_name} {mode}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MIDI: {out_midi}")
    return tonic_name, mode

# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Transcribe monophonic oud to notes + solfege + MIDI")
    p.add_argument("--audio", required=True, help="Path to WAV/MP3 of solo oud (short clip recommended)")
    p.add_argument("--out_csv", default="out/transcription.csv", help="Where to save the CSV")
    p.add_argument("--out_midi", default="out/transcription.mid", help="Where to save the MIDI")
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()

    run(args.audio, args.out_csv, args.out_midi, sr=args.sr)
    
