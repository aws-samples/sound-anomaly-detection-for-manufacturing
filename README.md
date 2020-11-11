# Performing anomaly detection on industrial equipment using audio signals

This repository contains a sample on how to perform anomaly detection on machine sounds (based on the [MIMII Dataset](https://zenodo.org/record/3384388)) leveraging several approaches.

**Running time:** *once the dataset is downloaded, it takes roughly an hour and a half to go through all these notebooks from start to finish.*

## Overview
Industrial companies have been collecting a massive amount of time series data about their operating processes, manufacturing production lines, industrial equipment... They sometime store years of data in historian systems. Whereas they are looking to prevent equipment breakdown that would stop a production line, avoid catastrophic failures in a power generation facility or improving their end product quality by adjusting their process parameters, having the ability to process time series data is a challenge that modern cloud technologies are up to. In this post, we are going to focus on preventing machine breakdown from happening.

In many cases, machine failures are tackled by either reactive action (stop the line and repair...) or costly preventive maintenance where you have to build the proper replacement parts inventory and schedule regular maintenance activities. Skilled machine operators are the most valuable assets in such settings: years of experience allow them to develop a fine knowledge of how the machinery should operate, they become expert listeners and are able to develop unusual behavior and sounds in rotating and moving machines. However, production lines are becoming more and more automated, and augmenting these machine operators with AI-generated insights is a way to maintain and develop the fine expertise needed to prevent industrials undergoing a reactive posture when dealing with machine breakdowns.

This is a companion repository for a blog post on AWS Machine Learning Blog, where we compare and contrast two different approaches to identify a malfunctioning machine for which we have sound recordings: we will start by building a neural network based on an autoencoder architecture and we will then use an image-based approach where we will feed “images from the sound” (namely spectrograms) to an image based automated ML classification feature.

### Installation instructions
[Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

Navigate to the SageMaker console and create a new instance. Using an **ml.c5.2xlarge instance** with a **25 GB attached EBS volume** is recommended to process the dataset comfortably

You need to ensure that this notebook instance has an IAM role which allows it to call the Amazon Rekognition Custom Labels API:
1. In your IAM console, look for the SageMaker execution role endorsed by your notebook instance (a role with a name like AmazonSageMaker-ExecutionRole-yyyymmddTHHMMSS)
2. Click on Attach Policies and look for this managed policy: AmazonRekognitionCustomLabelsFullAccess
3. Check the box next to it and click on Attach Policy

Your SageMaker notebook instance can now call the Rekognition Custom Labels APIs.

You can know navigate back to the Amazon SageMaker console, then to the Notebook Instances menu. Start your instance and launch either Jupyter or JupyterLab session. From there, you can launch a new terminal and clone this repository into your local development machine using `git clone`.

### Repository structure
Once you've cloned this repo, browse to the [data exploration](1_data_exploration.ipynb) notebook: this first notebook will download and prepare the data necessary for the other ones.

The dataset used is a subset of the MIMII dataset dedicated to industrial fans sound. This 10 GB archive will be downloaded in the /tmp directory: if you're using a SageMaker instance, you should have enough space on the ephemeral volume to download it. The unzipped data is around 15 GB large and will be located in the EBS volume, make sure it is large enough to prevent any out of space error.

```
.
|
+-- README.md                                 <-- This instruction file
|
+-- autoencoder/
|   |-- model.py                              <-- The training script used as an entrypoint of the 
|   |                                             TensorFlow container
|   \-- requirements.txt                      <-- Requirements file to update the training container 
|                                                 at launch
|
+-- pictures/                                 <-- Assets used in in the introduction and README.md
|
+-- tools/
|   |-- rekognition_tools.py                  <-- Utilities to manage Rekognition custom labels models
|   |                                             (start, stop, get inference...)
|   |-- sound_tools.py                        <-- Utilities to manage sounds dataset
|   \-- utils.py                              <-- Various tools to build files list, plot curves, and 
|                                                 confusion matrix... 
|
+-- 0_introduction.ipynb                      <-- Expose the context
|
+-- 1_data_exploration.ipynb                  <-- START HERE: data exploration notebook, useful to 
|                                                 generate the datasets, get familiar with sound datasets
|                                                 and basic frequency analysis
|
+-- 2_custom_autoencoder.ipynb                <-- Using the SageMaker TensorFlow container to build a 
|                                                 custom autoencoder
|
\-- 3_rekognition_custom_labels.ipynb         <-- Performing the same tasks by calling the Rekognition 
                                                  Custom Labels API
```

## Questions

Please contact [@michaelhoarau](https://twitter.com/michaelhoarau) or raise an issue on this repository.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
This collection of notebooks is licensed under the MIT-0 License. See the LICENSE file.
