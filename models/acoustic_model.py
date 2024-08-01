# acoustic_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, BatchNormalization, TimeDistributed

def build_acoustic_model(input_shape, num_classes=29):
    """Builds an acoustic model using a Bidirectional LSTM."""
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.5)(x)
    outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(x)
    
    model = Model(inputs, outputs)
    return model

def train_acoustic_model(model, X_train, y_train, X_val, y_val, config):
    """Trains the acoustic model."""
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=batch_size, epochs=epochs)
    return history

def save_acoustic_model(model, model_path):
    """Saves the trained acoustic model to a file."""
    model.save(model_path)
    
def load_acoustic_model(model_path, input_shape):
    """Loads a trained acoustic model from a file."""
    model = build_acoustic_model(input_shape)
    model.load_weights(model_path)
    return model
