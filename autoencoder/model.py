# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse, os
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

def autoencoder_model(input_dims):
    """
    Defines a Keras model for performing the anomaly detection. 
    This model is based on a simple dense autoencoder.
    
    PARAMS
    ======
        inputs_dims (integer) - number of dimensions of the input features
        
    RETURN
    ======
        Model (tf.keras.models.Model) - the Keras model of our autoencoder
    """
    
    # Autoencoder definition:
    inputLayer = Input(shape=(input_dims,))
    h = Dense(64, activation="relu")(inputLayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(input_dims, activation=None)(h)

    return Model(inputs=inputLayer, outputs=h)

def parse_arguments():
    """
    Parse the command line arguments passed when running this training script
    
    RETURN
    ======
        args (ArgumentParser) - an ArgumentParser instance command line arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--frame', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()
    
    return args
    
def train(training_dir, model_dir, n_mels, frame, lr, batch_size, epochs, gpu_count):
    """
    Main training function.
    
    PARAMS
    ======
        training_dir (string) - location where the training data are
        model_dir (string) - location where to store the model artifacts
        n_mels (integer) - number of Mel buckets to build the spectrograms
        frames (integer) - number of sliding windows to use to slice the Mel spectrogram
        lr (float) - learning rate
        batch_size (integer) - batch size
        epochs (integer) - number of epochs
        gpu_count (integer) - number of GPU to distribute the job on
    """
    # Load training data:
    train_data_file = os.path.join(training_dir, 'train_data.pkl')
    with open(train_data_file, 'rb') as f:
        train_data = pickle.load(f) 
    
    # Builds the model:
    model = autoencoder_model(n_mels * frame)
    print(model.summary())
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    # Model preparation:
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    # Model training: this is an autoencoder, we 
    # use the same data for training and validation:
    history = model.fit(
        train_data, 
        train_data,
        batch_size=batch_size,
        validation_split=0.1,
        epochs=epochs,
        shuffle=True,
        verbose=2
    )
    
    # Save the trained model:
    os.makedirs(os.path.join(model_dir, 'model/1'), exist_ok=True)
    model.save(os.path.join(model_dir, 'model/1'))

if __name__ == '__main__':
    # Initialization:
    tf.random.set_seed(42)
    
    # Parsing command line arguments:
    args = parse_arguments()
    epochs       = args.epochs
    n_mels       = args.n_mels
    frame        = args.frame
    lr           = args.learning_rate
    batch_size   = args.batch_size
    gpu_count    = args.gpu_count
    model_dir    = args.model_dir
    training_dir = args.training
    
    # Launch the training:
    train(training_dir, model_dir, n_mels, frame, lr, batch_size, epochs, gpu_count)