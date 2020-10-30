# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import sys
import librosa
import librosa.display
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_sound_file(wav_name, mono=False, channel=0):
    """
    Loads a sound file
    
    PARAMS
    ======
        wav_name (string) - location to the WAV file to open
        mono (boolean) - signal is in mono (if True) or Stereo (False, default)
        channel (integer) - which channel to load (default to 0)
    
    RETURNS
    =======
        signal (numpy array) - sound signal
        sampling_rate (float) - sampling rate detected in the file
    """
    multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    signal = np.array(multi_channel_data)[channel, :]
    
    return signal, sampling_rate

def get_magnitude_scale(file, n_fft=1024, hop_length=512):
    """
    Get the magnitude scale from a wav file.
    
    PARAMS
    ======
        file (string) - filepath to the location of the WAV file
        n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
        hop_length (integer) - window increment when computing STFT

    RETURNS
    =======
        dB (ndarray) - returns the log scaled amplitude of the sound file
    """
    # Load the sound data:
    signal, sampling_rate = load_sound_file(file)

    # Compute the short-time Fourier transform of the signal:
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # Map the magnitude to a decibel scale:
    dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    return dB

def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    """
    Extract features from a sound signal, given a sampling rate sr. This function 
    computes the Mel spectrogram in log scales (getting the power of the signal).
    Then we build N frames (where N = frames passed as an argument to this function):
    each frame is a sliding window in the temporal dimension.
    
    PARAMS
    ======
        signal (array of floats) - numpy array as returned by load_sound_file()
        sr (integer) - sampling rate of the signal
        n_mels (integer) - number of Mel buckets (default: 64)
        frames (integer) - number of sliding windows to use to slice the Mel spectrogram
        n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
        hop_length (integer) - window increment when computing STFT
    """
    
    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    
    # Skips short signals:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)
    
    # Build N sliding windows (=frames) and concatenate them to build a feature vector:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
        
    return features

def generate_dataset(files_list, n_mels=64, frames=5, n_fft=1024, hop_length=512):
    """
    Takes a list for WAV files as an input and generate a numpy array with
    the extracted features.
    
    PARAMS
    ======
        files_list (list) - list of files to generate a dataset from
        n_mels (integer) - number of Mel buckets (default: 64)
        frames (integer) - number of sliding windows to use to slice the Mel 
                           spectrogram
        n_fft (integer) - length of the windowed signal to compute the short 
                          Fourier transform on
        hop_length (integer) - window increment when computing STFT
        
    RETURNS
    =======
        dataset (numpy array) - dataset
    """
    # Number of dimensions for each frame:
    dims = n_mels * frames
    
    for index in tqdm(range(len(files_list)), desc='Extracting features'):
        # Load signal
        signal, sr = load_sound_file(files_list[index])
        
        # Extract features from this signal:
        features = extract_signal_features(
            signal, 
            sr, 
            n_mels=n_mels, 
            frames=frames, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
            
        dataset[features.shape[0] * index : features.shape[0] * (index + 1), :] = features

    return dataset

def scale_minmax(X, min=0.0, max=1.0):
    """
    Minmax scaler for a numpy array
    
    PARAMS
    ======
        X (numpy array) - array to scale
        min (float) - minimum value of the scaling range (default: 0.0)
        max (float) - maximum value of the scaling range (default: 1.0)
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def generate_spectrograms(list_files, output_dir, n_mels=64, n_fft=1024, hop_length=512):
    """
    Generate spectrograms pictures from a list of WAV files. Each sound
    file in WAV format is processed to generate a spectrogram that will 
    be saved as a PNG file.
    
    PARAMS
    ======
        list_files (list) - list of WAV files to process
        output_dir (string) - root directory to save the spectrogram to
        n_mels (integer) - number of Mel buckets (default: 64)
        n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
        hop_length (integer) - window increment when computing STFT
        
    RETURNS
    =======
        files (list) - list of spectrogram files (PNG format)
    """
    files = []
    
    # Loops through all files:
    for index in tqdm(range(len(list_files)), desc=f'Building spectrograms for {output_dir}'):
        # Building file name for the spectrogram PNG picture:
        file = list_files[index]
        path_components = file.split('/')
        
        # machine_id = id_00, id_02...
        # sound_type = normal or abnormal
        # wav_file is the name of the original sound file without the .wav extension
        machine_id, sound_type = path_components[-3], path_components[-2]
        wav_file = path_components[-1].split('.')[0]
        filename = sound_type + '-' + machine_id + '-' + wav_file + '.png'
        
        # Example: train/normal/normal-id_02-00000259.png:
        filename = os.path.join(output_dir, sound_type, filename)

        if not os.path.exists(filename):
            # Loading sound file and generate Mel spectrogram:
            signal, sr = load_sound_file(file)
            mels = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            mels = librosa.power_to_db(mels, ref=np.max)

            # Preprocess the image: min-max, putting 
            # low frequency at bottom and inverting to 
            # match higher energy with black pixels:
            img = scale_minmax(mels, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img
            img = Image.fromarray(img)

            # Saving the picture generated to disk:
            img.save(filename)

        files.append(filename)
        
    return files
