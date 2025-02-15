import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import urllib.request
import argparse
import json  # Import json module


#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Download class labels
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
LABELS_PATH = "yamnet_class_map.csv"
urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)

# Read class labels
class_labels = np.array(open(LABELS_PATH).read().splitlines())

# Short, Pro Tools-friendly instrument names including keyboards, drums, and vocals
INSTRUMENT_SHORT_NAMES = {
    # Drums & Percussion
    "Drum": "Drum Loop",
    "Percussion": "Perc Loop",
    "Cymbal": "Cymbal Hit",
    "Snare drum": "Snare",
    "Bass drum": "Kick Drum",
    "Hi-hat": "Hi-hat",
    "Tom": "Tom",
    "Floor tom": "Floor Tom",
    "Crash cymbal": "Crash Cymbal",
    "Ride cymbal": "Ride Cymbal",
    "Timpani": "Timpani",
    "Congas": "Congas",
    "Bongo": "Bongo",
    "Maracas": "Maracas",
    "Shaker": "Shaker",
    "Claves": "Claves",
    "Tambourine": "Tambourine",
    
    # Keyboards
    "Piano": "Piano Chords",
    "Electric piano": "Electric Piano",
    "Organ": "Organ",
    "Synthesizer": "Synth Lead",
    "Harpsichord": "Harpsichord",
    "Clavinet": "Clavinet",
    "Accordion": "Accordion",
    "Keytar": "Keytar",
    
    # Vocals
    "Male speech": "Male Voice",
    "Female speech": "Female Voice",
    "Male singing": "Male Vocal",
    "Female singing": "Female Vocal",
    "Singing": "Singing",
    "Singing voice": "Vocal",
    "Choir": "Choir",
    "Male choir": "Male Choir",
    "Female choir": "Female Choir",
    "Vocalization": "Vocalization",

    # Strings
    "Guitar": "Guitar Riff",
    "Electric guitar": "Electric Guitar",
    "Acoustic guitar": "Acoustic Guitar",
    "Violin": "Violin",
    "Cello": "Cello",

    # Brass
    "Trumpet": "Trumpet Stab",
    "Saxophone": "Sax Riff",
    "Flute": "Flute Melody",
    "Clarinet": "Clarinet Line",
}

def classify_audio(file_paths, confidence_threshold=0.5):
    track_names = []
    
    for file_path in file_paths:
        # Load audio file
        waveform, sr = librosa.load(file_path, sr=16000)  # Resample to 16kHz

        # Run model and get scores
        scores, embeddings, spectrogram = yamnet_model(waveform)

        # Get the average score across the time frames for each class
        mean_scores = np.mean(scores.numpy(), axis=0)
        
        # Get top predictions sorted by score
        sorted_indices = np.argsort(mean_scores)[::-1]
        
        # Check if the highest score exceeds the confidence threshold
        if mean_scores[sorted_indices[0]] < confidence_threshold:
            track_names.append("Unknown Sound")  # Below confidence threshold
            continue

        # Find the first category that matches an instrument
        for index in sorted_indices:
            label = class_labels[index]
            for key in INSTRUMENT_SHORT_NAMES.keys():
                if key in label:
                    track_names.append(INSTRUMENT_SHORT_NAMES[key])
                    break
            else:
                continue
            break
        else:
            track_names.append("Unknown Sound")
    
    return track_names

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Classify audio files using YAMNet.")
    parser.add_argument("files", metavar="F", type=str, nargs="+", help="List of audio file paths to classify.")
    args = parser.parse_args()

    # Classify the audio files and output the result as JSON
    track_names = classify_audio(args.files, confidence_threshold=0.5)
    
    # Print the result as a JSON array
    print(json.dumps(track_names))
