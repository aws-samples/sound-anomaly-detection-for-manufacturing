# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
from tqdm import tqdm

def md5(fname):
    """
    This function builds an MD5 hash for the file passed as argument.
    
    PARAMS
    ======
        fname (string)
            Full path and filename
            
    RETURNS
    =======
        hash (string)
            The MD5 hash of the file
    """
    filesize = os.stat(fname).st_size
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(4096), b""), total=filesize/4096):
            hash_md5.update(chunk)
            
    return hash_md5.hexdigest()

def build_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
    """
    Generate a list of files located in the root dir.
    
    PARAMS
    ======
        root_dir (string)
            Root directory to walk
        abnormal_dir (string)
            Directory where the abnormal files are located. 
            Defaults to 'abnormal'
        normal_dir (string)
            Directory where the normal files are located.
            Defaults to 'normal'

    RETURNS
    =======
        normal_files (list)
            List of files in the normal directories
        abnormal_files (list)
            List of files in the abnormal directories
    """
    normal_files = []
    abnormal_files = []
    
    # Loops through the directories to build a normal and an abnormal files list:
    for root, dirs, files in os.walk(top = os.path.join(root_dir)):
        for name in files:
            current_dir_type = root.split('/')[-1]
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))
                
    return normal_files, abnormal_files


def generate_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
    """
    Generate a list of files located in the root dir and sort test and train 
    files and labels to be used by an autoencoder. This means that the train 
    set only contains normal values, whereas the test set is balanced between 
    both types.
    
    PARAMS
    ======
        root_dir (string)
            Root directory to walk
        abnormal_dir (string)
            Directory where the abnormal files are located. 
            Defaults to 'abnormal'
        normal_dir (string)
            Directory where the normal files are located.
            Defaults to 'normal'
            
    RETURNS
    =======
        train_files (list)
            List of files to train with (only normal data)
        train_labels (list)
            List of labels (0s for normal)
        test_files (list)
            Balanced list of files with both normal and abnormal data
        test_labels (list)
            List of labels (0s for normal and 1s otherwise)
    """
    normal_files = []
    abnormal_files = []
    
    # Loops through the directories to build a normal and an abnormal files list:
    for root, dirs, files in os.walk(top = os.path.join(root_dir)):
        for name in files:
            current_dir_type = root.split('/')[-1]
            if current_dir_type == abnormal_dir:
                abnormal_files.append(os.path.join(root, name))
            if current_dir_type == normal_dir:
                normal_files.append(os.path.join(root, name))

    # Shuffle the normal files in place:
    random.shuffle(normal_files)

    # The test files contains all the abnormal files and the same number of normal files:
    test_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    test_labels = np.concatenate((np.zeros(len(abnormal_files)), np.ones(len(abnormal_files))), axis=0)
    
    # The train files contains all the remaining normal files:
    train_files = normal_files[len(abnormal_files):]
    train_labels = np.zeros(len(train_files))
    
    return train_files, train_labels, test_files, test_labels

def generate_error_types(df, ground_truth_col='Ground Truth', prediction_col='Prediction', normal_label=0.0, anomaly_label=1.0):
    """
    Compute false positive and false negatives columns based on the prediction
    and ground truth columns from a dataframe.
    
    PARAMS
    ======
        df (dataframe)
            Dataframe where the ground truth and prediction columns are available
        ground_truth_col (string)
            Column name for the ground truth values. Defaults to "Ground Truth"
        prediction_col (string)
            Column name for the predictied values. Defaults to "Prediction"
        normal_label (object)
            Value taken by a normal value. Defaults to 0.0
        anomaly_label (object)
            Value taken by an abnormal value. Defaults to 1.0
            
    RETURNS
    =======
        df (dataframe)
            An updated dataframe with 4 new binary columns for TP, TN, FP and FN.
    """
    df['TP'] = 0
    df['TN'] = 0
    df['FP'] = 0
    df['FN'] = 0
    df.loc[(df[ground_truth_col] == df[prediction_col]) & (df[ground_truth_col] == normal_label), 'TP'] = 1
    df.loc[(df[ground_truth_col] == df[prediction_col]) & (df[ground_truth_col] == anomaly_label), 'TN'] = 1
    df.loc[(df[ground_truth_col] != df[prediction_col]) & (df[ground_truth_col] == normal_label), 'FP'] = 1
    df.loc[(df[ground_truth_col] != df[prediction_col]) & (df[ground_truth_col] == anomaly_label), 'FN'] = 1
    
    return df

def plot_curves(FP, FN, nb_samples, threshold_min, threshold_max, threshold_step):
    """
    Plot the false positives and false negative samples number depending on a given threshold.
    
    PARAMS
    ======
        FP (dataframe)
            Number of false positives depending on the threshold
        FN (dataframe)
            Number of false negatives depending on the threshold
        threshold_min (float)
            Minimum threshold to plot for
        threshold_max (float)
            Maximum threshold to plot for
        threshold_step (float)
            Threshold step to plot these curves
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    min_FN = np.argmin(FN)
    min_FP = np.where(FP == np.min(FP))[0][-1]
    plot_top = max(FP + FN) + 1

    # Grid customization:
    major_ticks = np.arange(threshold_min, threshold_max, 1.0 * threshold_step)
    minor_ticks = np.arange(threshold_min, threshold_max, 0.2 * threshold_step)
    ax.set_xticks(major_ticks);
    ax.set_xticks(minor_ticks, minor=True);
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0, linewidth=1.0)
    
    # Plot false positives and false negatives curves
    plt.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FP, label='False positive', color='tab:red')
    plt.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FN, label='False negative', color='tab:green')

    # Finalize the plot with labels and legend:
    plt.xlabel('Reconstruction error threshold (%)', fontsize=16)
    plt.ylabel('# Samples', fontsize=16)
    plt.legend()
    
def print_confusion_matrix(confusion_matrix, class_names, figsize = (4,3), fontsize=14):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix,
    as a heatmap.
    
    PARAMS
    ======
        confusion_matrix (numpy.ndarray)
            The numpy.ndarray object returned from a call to 
            sklearn.metrics.confusion_matrix. Similarly constructed 
            ndarrays can also be used.
        class_names (list)
            An ordered list of class names, in the order they index the given
            confusion matrix.
        figsize (tuple)
            A 2-long tuple, the first value determining the horizontal size of
            the ouputted figure, the second determining the vertical size.
            Defaults to (10,7).
        fontsize: (int)
            Font size for axes labels. Defaults to 14.
        
    RETURNS
    =======
        matplotlib.figure.Figure: The resulting confusion matrix figure
    """
    # Build a dataframe from the confusion matrix passed as argument:
    df_cm = pd.DataFrame(confusion_matrix, 
                         index=class_names, 
                         columns=class_names)
    
    # Plot the confusion matrix:
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap='viridis')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    # Figure customization:
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    
    return fig