# train.py

import yaml
from sklearn.model_selection import train_test_split
from models.acoustic_model import build_acoustic_model, train_acoustic_model, save_acoustic_model
from models.language_model import build_language_model, train_language_model, save_language_model
from utils.audio_processing import load_audio_data, preprocess_audio

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Load and preprocess data
    audio_data = load_audio_data(config['data']['raw_data_path'], config['data']['sample_rate'])
    X, y = preprocess_audio(audio_data, config['data'])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Acoustic Model
    print("Training Acoustic Model...")
    acoustic_model = build_acoustic_model(input_shape=(None, config['data']['n_mfcc']))
    train_acoustic_model(acoustic_model, X_train, y_train, X_val, y_val, config['model'])
    save_acoustic_model(acoustic_model, config['model']['acoustic_model_path'])

    # Train Language Model
    print("Training Language Model...")
    language_model = build_language_model()
    train_language_model(language_model, X_train, y_train, X_val, y_val, config['model'])
    save_language_model(language_model, config['model']['language_model_path'])

    print("Training complete.")

if __name__ == "__main__":
    main()
