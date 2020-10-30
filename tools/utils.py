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
    filesize = os.stat(fname).st_size
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in tqdm(iter(lambda: f.read(4096), b""), total=filesize/4096):
            hash_md5.update(chunk)
            
    return hash_md5.hexdigest()

def build_files_list(root_dir, abnormal_dir='abnormal', normal_dir='normal'):
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
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    min_FN = np.argmin(FN)# * step
    min_FP = np.where(FP == np.min(FP))[0][-1]# * step
    plot_top = max(FP + FN) + 1

    major_ticks = np.arange(threshold_min, threshold_max, 1.0 * threshold_step)
    minor_ticks = np.arange(threshold_min, threshold_max, 0.2 * threshold_step)
    ax.set_xticks(major_ticks);
    ax.set_xticks(minor_ticks, minor=True);
    
    plt.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FP, label='False positive', color='tab:red')
    plt.plot(np.arange(threshold_min, threshold_max + threshold_step, threshold_step), FN, label='False negative', color='tab:green')
    
    major_ticks = np.arange(0, plot_top, int(0.2 * max(FP)))
    minor_ticks = np.arange(0, plot_top, int(0.05 * max(FP)))
    #ax.set_yticks(major_ticks);
    #ax.set_yticks(minor_ticks, minor=True);
    
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1.0, linewidth=1.0)

    plt.xlabel('Reconstruction error threshold (%)', fontsize=16)
    plt.ylabel('# Samples', fontsize=16)
    plt.legend()
    
def print_confusion_matrix(confusion_matrix, class_names, figsize = (4,3), fontsize=14):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    PARAMS
    ======
        confusion_matrix (numpy.ndarray)
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
            Similarly constructed ndarrays can also be used.
        class_names (list)
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize (tuple)
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: (int)
            Font size for axes labels. Defaults to 14.
        
    RETURNS
    =======
        matplotlib.figure.Figure: The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap='viridis')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    
    return fig