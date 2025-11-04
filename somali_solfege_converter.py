import pretty_midi
import pandas as pd

# ======== CONFIGURATION ========
# Path to your extracted MIDI file
midi_path = "qaraami_extracted.mid"

# Somali pentatonic scale (approximated in C major reference)
# Do, Re, Mi, So, La → C, D, E, G, A
somali_scale = [0, 2, 4, 7, 9]
somali_solfege = ["Do", "Re", "Mi", "So", "La"]

# ======== FUNCTIONS ========

def nearest_pentatonic(note_num):
    """Quantize a MIDI note number to the nearest Somali pentatonic scale note"""
    octave = note_num // 12
    base = note_num % 12
    diffs = [(abs(base - s), s) for s in somali_scale]
    _, closest = min(diffs, key=lambda x: x[0])
    return (octave * 12) + closest

def note_to_solfege(note_num):
    """Convert MIDI note to Somali-style solfege"""
    return somali_solfege[somali_scale.index(nearest_pentatonic(note_num) % 12)]

# ======== MAIN PROCESS ========
midi_data = pretty_midi.PrettyMIDI(midi_path)
rows = []

for instrument in midi_data.instruments:
    for note in instrument.notes:
        somali_pitch = nearest_pentatonic(note.pitch)
        solfege_name = note_to_solfege(note.pitch)
        rows.append({
            "Start (s)": round(note.start, 3),
            "End (s)": round(note.end, 3),
            "MIDI_Pitch": note.pitch,
            "Quantized_Pitch": somali_pitch,
            "Solfege": solfege_name,
            "Western_Note": pretty_midi.note_number_to_name(note.pitch),
            "Quantized_Note": pretty_midi.note_number_to_name(somali_pitch)
        })

# ======== SAVE TO CSV ========
df = pd.DataFrame(rows)
csv_path = "somali_solfege_output.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Somali solfege CSV created: {csv_path}")
print(df.head(10))
