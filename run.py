import yaml
import os
from models.acoustic_model import build_acoustic_model, train_acoustic_model
from models.language_model import build_language_model, train_language_model
from utils.audio_processing import load_audio_data, preprocess_audio
from utils.evaluation import evaluate_model

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']
    sample_rate = config['data']['sample_rate']
    max_audio_duration = config['data']['max_audio_duration']
    
    print("Loading and preprocessing audio data...")
    audio_data = load_audio_data(raw_data_path, sample_rate, max_audio_duration)
    preprocessed_data = preprocess_audio(audio_data, config['data'])

    # Build and train models
    print("Building and training acoustic model...")
    acoustic_model = build_acoustic_model(input_shape=(None, config['data']['n_mfcc']))
    train_acoustic_model(acoustic_model, preprocessed_data, config['model'])

    print("Building and training language model...")
    language_model = build_language_model()
    train_language_model(language_model, config['model'])

    # Evaluate models
    print("Evaluating models...")
    test_samples_path = config['data']['test_samples_path']
    test_data = load_audio_data(test_samples_path, sample_rate, max_audio_duration)
    evaluate_model(acoustic_model, language_model, test_data, config['evaluation'])

    print("All tasks completed.")

if __name__ == "__main__":
    main()
