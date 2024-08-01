# feature_extraction.py

import librosa
import numpy as np

def extract_mfcc(y, sr, n_mfcc=13):
    """Extract MFCC features from an audio signal."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_chroma(y, sr, n_chroma=12):
    """Extract chroma features from an audio signal."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    return chroma

def extract_spectral_contrast(y, sr, n_bands=6):
    """Extract spectral contrast from an audio signal."""
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands)
    return spectral_contrast

def extract_features(y, sr, n_mfcc=13, n_chroma=12, n_bands=6):
    """Extract multiple features from an audio signal."""
    mfcc = extract_mfcc(y, sr, n_mfcc)
    chroma = extract_chroma(y, sr, n_chroma)
    spectral_contrast = extract_spectral_contrast(y, sr, n_bands)
    return mfcc, chroma, spectral_contrast
