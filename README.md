# BackPropagation-Algorithm

# Backpropagation in CNNs for Audio Classification

## Overview
This project shows training a Convolutional Neural Network (CNN) using backpropagation to classify audio. Audio signals are first converted to Mel-spectrograms to be used as input for the model.

## Objective
In this project i will explain and implement the backpropagation algorithm using a deep learning framework to classify audio files.

## Dataset
The dataset consists of:
•⁠  ⁠Heart sounds (HS)
•⁠  ⁠Lung sounds (LS)
•⁠  ⁠Mixed sounds (Mix)

These are stored as ⁠ .wav ⁠ files and come with a corresponding metadata CSV file.

### Dataset Source
Our dataset is based on some public biomedical audio datasets that are used for heart and lung sounds classification in many researches and educational works.

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope


### Dataset Structure
ML_project/
├── HS/
├── LS/
├── Mix/
├── HS.csv
├── LS.csv
└── Mix.csv

## Why the Dataset is Not Included
The dataset is not included in this repository because:
•⁠  ⁠Audio files are large and exceed GitHub size limits
•⁠  ⁠GitHub restricts files larger than 100MB
•⁠  ⁠Including large datasets reduces repository performance and usability

This is a skeleton project. What this means is that this folder structure has been created so that you can copy this project.

## Where is Backpropagation Shown?

Backpropagation is implemented in two key parts of the project:

1. Model Training

   ```python
   model.fit(...)
   ``` 
TensorFlow automatically performs:

•Forward pass
•Loss computation
•Backward pass (backpropagation)
•Weight updates using gradients

2. Explicit Gradient Computation, Using:
 ```python
tf.GradientTape()
 ```
This section computes explicitly the gradients of the loss function w.r.t. to the model parameters. It is a way to illustrate internally the workings of backpropagation. 

## Reproducibility

The project is designed to be fully reproducible:

Clear dataset structure is provided
Code runs end-to-end without modification
All dependencies are listed in requirements.txt


## Installation
```bash
pip install -r requirements.txt

