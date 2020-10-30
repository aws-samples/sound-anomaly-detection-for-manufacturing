# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import boto3
import json
import pandas as pd
import s3fs
import utils

from datetime import datetime

def create_manifest_from_bucket(bucket, prefix, folder, labels, output_bucket):
    """
    Based on a bucket / prefix location on S3, this function will crawl this 
    location for images and generate a JSON manifest file compatible with 
    Rekognition Custom Labels.
    
    PARAMS
    ======
        bucket (string) - bucket name
        prefix (string) - S3 prefix where to look for the images
        folder (string) - either train or test
        labels (list) - list of labels to look for (normal, anomaly)
        output_bucket (string) - where to upload the JSON manifest file to
    """
    # Get a creation date:
    creation_date = str(pd.to_datetime(datetime.now()))[:23].replace(' ','T')
    
    # Assign a distinct identifier for each label:
    auto_label = {}
    for index, label in enumerate(labels):
        auto_label.update({label: index + 1})
    
    # Get a handle on an S3 filesystem object:
    fs = s3fs.S3FileSystem()
    
    # Create a manifest file in the output directory passed as argument:
    with fs.open(output_bucket + f'/{folder}.manifest', 'w') as f:
        # We expect one subfolder for each label:
        for label in labels:
            # Loops through each file present at this point:
            for file in fs.ls(path=f'{bucket}/{prefix}/{folder}/{label}/', detail=True):
                # We only care for files, not directories:
                if file['Size'] > 0:
                    key = file['Key']
                    
                    # Build a Ground Truth format manifest row:
                    manifest_row = {
                        'source-ref': f's3://{key}',
                        'auto-label': auto_label[label],
                        'auto-label-metadata': {
                            'confidence': 1,
                            'job-name': 'labeling-job/auto-label',
                            'class-name': label,
                            'human-annotated': 'yes',
                            'creation-date': creation_date,
                            'type': 'groundtruth/image-classification'
                        }
                    }

                    # Write this line to the manifest:
                    f.write(json.dumps(manifest_row, indent=None) + '\n')
                    
def start_model(project_arn, model_arn, version_name, min_inference_units=1):
    """
    Start a Rekognition Custom Labels model.
    
    PARAMS
    ======
        project_arn (string) - project ARN
        model_arn (string) - project version ARN
        version_name (string) - project version name
        min_inference_units (integer) - inference unit to provision for the 
                                        endpoint which will be deployed for 
                                        this particular project version.
    """
    client = boto3.client('rekognition')

    try:
        # Start the model
        print('Starting model: ' + model_arn)
        response = client.start_project_version(ProjectVersionArn=model_arn, MinInferenceUnits=min_inference_units)
        
        # Wait for the model to be in the running state:
        project_version_running_waiter = client.get_waiter('project_version_running')
        project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])

        # Get the running status
        describe_response=client.describe_project_versions(ProjectArn=project_arn, VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage'])
            
    except Exception as e:
        print(e)
        
    print('Done.')
    
def stop_model(model_arn):
    """
    Stops a Rekognition Custom Labels model.
    
    PARAMS
    ======
        model_arn (string) - project version ARN
    """
    print('Stopping model:' + model_arn)

    # Stop the model:
    try:
        reko = boto3.client('rekognition')
        response = reko.stop_project_version(ProjectVersionArn=model_arn)
        status = response['Status']
        print('Status: ' + status)
        
    except Exception as e:  
        print(e)  

    print('Done.')
    
def show_custom_labels(model, bucket, image, min_confidence):
    """
    Calls the Rekognition detect_custom_labels() API to get the prediction for
    a given image.
    
    PARAMS
    ======
        model (string) - project version ARN
        bucket (string) - bucket where the image is located
        image (string) - complete S3 prefix where the image is located
        min_confidence (float) - minimum confidence score to return a result
        
    RETURNS
    =======
        Returns the custom label response
    """
    # Call DetectCustomLabels from the Rekognition API: this will give us the list 
    # of labels detected for this picture and their associated confidence level:
    reko = boto3.client('rekognition')
    try:
        response = reko.detect_custom_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': image}},
            MinConfidence=min_confidence,
            ProjectVersionArn=model
        )
        
    except Exception as e:
        print(f'Exception encountered when processing {image}')
        print(e)
        
    # Returns the list of custom labels for the image passed as an argument:
    return response['CustomLabels']

def get_results(project_version_arn, bucket, s3_path, label=None, verbose=True):
    """
    Sends a list of pictures located in an S3 path to
    the endpoint to get the associated predictions.
    
    PARAMS
    ======
        project_version_arn (string) - ARN of the model to query
        bucket (string) - bucket name
        s3_path (string) - prefix where to look the images for
        label (string) - ground truth label of the images
        verbose (boolean) - shows a progress bar if True (defaults to True)
        
    RETURNS
    =======
        predictions (dataframe)
            A dataframe with the following columns: image, 
            abnormal probability, normal probability and 
            ground truth.
    """

    fs = s3fs.S3FileSystem()
    data = {}
    counter = 0
    predictions = pd.DataFrame(columns=['image', 'normal', 'abnormal'])
    
    for file in fs.ls(path=s3_path, detail=True, refresh=True):
        if file['Size'] > 0:
            image = '/'.join(file['Key'].split('/')[1:])
            if verbose == True: print('.', end='')

            labels = show_custom_labels(project_version_arn, bucket, image, 0.0)
            for L in labels:
                data[L['Name']] = L['Confidence']
                
            predictions = predictions.append(pd.Series({
                'image': file['Key'].split('/')[-1],
                'abnormal': data['abnormal'],
                'normal': data['normal'],
                'ground truth': label
            }), ignore_index=True)
            
            # Temporization to prevent any throttling:
            counter += 1
            if counter % 100 == 0:
                if verbose == True: print('|', end='')
                time.sleep(1)
            
    return predictions

def reshape_results(df, unknown_threshold=50.0):
    """
    Reshape a results dataframe containing image path, normal and abnormal
    confidence levels into a more straightforward one with ground truth, 
    prediction and confidence level associated to each image.
    
    PARAMS
    ======
        df (dataframe)
            Input dataframe with the following columns: image, ground 
            truth, normal and abnormal.
            
        unknown_threshold (float)
            If a probability is lower than this threshold, we select 
            the other result (defaults to 50.0).
    """
    new_val_predictions = pd.DataFrame(columns=['Image', 'Ground Truth', 'Prediction', 'Confidence Level'])

    for index, row in df.iterrows():
        new_row = pd.Series(dtype='object')
        new_row['Image'] = row['image']
        new_row['Ground Truth'] = row['ground truth']
        if row['normal'] >= unknown_threshold:
            new_row['Prediction'] = 'normal'
            new_row['Confidence Level'] = row['normal'] / 100

        elif row['abnormal'] >= unknown_threshold:
            new_row['Prediction'] = 'abnormal'
            new_row['Confidence Level'] = row['abnormal'] / 100

        else:
            new_row['Prediction'] = 'unknown'
            new_row['Confidence Level'] = 0.0

        new_val_predictions = new_val_predictions.append(pd.Series(new_row), ignore_index=True)

    return new_val_predictions

def classification_report(input_df):
    """
    Generates a classification report (similar to what Amazon Rekognition 
    Custom Labels shows in the console) based on the input_df dataframe.
    
    PARAMS
    ======
    
    RETURNS
    =======
        performance (Pandas Series)
            Returns a Pandas series with the following attributes:
            - Label name: 'normal'
            - F1 score
            - Number of test images
            - Precision score
            - Recall score
            - Assumed threshold (computed as the confidence level minimum)
    """
    input_df = utils.generate_error_types(input_df, normal_label='normal', anomaly_label='abnormal')
    
    # Abnormal samples:
    df = input_df[input_df['Ground Truth'] == 'abnormal']
    TP = df['TN'].sum()
    FN = df['FN'].sum()
    FP = df['FP'].sum()
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    min_confidence_level = df.sort_values(by='Confidence Level', ascending=True).iloc[0]['Confidence Level']

    performance = pd.DataFrame(columns=['Label name', 'F1 score', 'Test images', 'Precision', 'Recall', 'Assumed threshold'])
    performance = performance.append(pd.Series({
        'Label name': 'abnormal',
        'F1 score': round(f1_score, 3),
        'Test images': input_df[input_df['Ground Truth'] == 'abnormal'].shape[0],
        'Precision': precision,
        'Recall': recall,
        'Assumed threshold': round(min_confidence_level,3)
    }), ignore_index=True)

    # Normal samples:
    df = input_df[input_df['Ground Truth'] == 'normal']
    TP = df['TP'].sum()
    FN = df['FN'].sum()
    FP = df['FP'].sum()
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    min_confidence_level = df.sort_values(by='Confidence Level', ascending=True).iloc[0]['Confidence Level']

    performance = performance.append(pd.Series({
        'Label name': 'normal',
        'F1 score': round(f1_score,3),
        'Test images': input_df[input_df['Ground Truth'] == 'normal'].shape[0],
        'Precision': precision,
        'Recall': recall,
        'Assumed threshold': round(min_confidence_level,3)
    }), ignore_index=True)

    return performance