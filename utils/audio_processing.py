# audio_processing.py

import librosa
import numpy as np
import os

def load_audio_file(file_path, sr=16000):
    """Load an audio file and return the audio time series and sampling rate."""
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def normalize_audio(y):
    """Normalize the audio signal to the range [-1, 1]."""
    return y / np.max(np.abs(y))

def load_audio_data(data_path, sr=16000):
    """Load all audio files in a directory and return a list of audio data and sampling rates."""
    audio_data = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_path, file_name)
            y, sr = load_audio_file(file_path, sr)
            y = normalize_audio(y)
            audio_data.append((y, sr))
    return audio_data

def preprocess_audio(audio_data, config):
    """Preprocess audio data: extract features (e.g., MFCCs) and format for model input."""
    n_mfcc = config.get('n_mfcc', 13)
    max_duration = config.get('max_audio_duration', 10.0)
    sample_rate = config.get('sample_rate', 16000)
    
    X = []
    y = []
    for y_data, sr in audio_data:
        mfccs = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=n_mfcc)
        mfccs = np.pad(mfccs, ((0, 0), (0, max_duration * sr // 512 - mfccs.shape[1])), mode='constant')
        X.append(mfccs)
        
        # Placeholder for target labels, needs to be replaced with actual labels
        y.append([0])  # Example: [0] for silence, [1] for speech

    X = np.array(X)
    y = np.array(y)
    return X, y
