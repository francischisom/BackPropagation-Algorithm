# Backpropagation in CNNs Using Heart/Lung Audio Classification

## Overview
This project shows training a Convolutional Neural Network (CNN) using backpropagation to classify audio. This audio signals are first converted to Mel-spectrograms to be used as input for the model.

## Learning Objectives
- Understand the role of backpropagation in deep learning
- Explore how CNNs learn from spectrogram-based audio features
- Implement and train a CNN using TensorFlow
- Interpret training results and model performance

## Dataset Description

The dataset consists of three categories of audio signals:
- Heart sounds (HS)
- Lung sounds (LS)
- Mixed sounds (Mix)

Each category is accompanied by a metadata CSV file containing identifiers for the corresponding audio files.

### Data Representation

Rather than feeding raw audio signals directly into the neural network, each audio file is converted into a Mel-spectrogram. This transformation converts time-series audio into a two-dimensional representation of frequency content over time.

This approach enables the use of Convolutional Neural Networks, which are highly effective for extracting spatial features from image-like inputs.

### Dataset Source
The dataset is based on some public biomedical audio datasets that are used for heart and lung sounds classification in many researches and educational works.

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


## Where Backpropagation is Shown

Backpropagation is implemented in two key parts of the project:

1. Model Training:

   ```python
   history = model.fit(...)
   ``` 
TensorFlow automatically performs:

•Forward pass
•Loss computation
•Backward pass (backpropagation)
•Weight updates using gradients


2. Through Explicit Gradient Computation: 
   
 ```python
with tf.GradientTape() as tape:
    predictions = model(x_batch, training=True)
    loss_value = loss_fn(y_batch, predictions)

gradients = tape.gradient(loss_value, model.trainable_variables)
 ```
This section computes explicitly the gradients of the loss function with respect to the model parameters. It is a way to illustrate internally the workings of backpropagation. 

## Project Workflow

Audio Files → Preprocessing → Mel-Spectrogram Extraction → CNN Model → Training with Backpropagation → Evaluation

## Running on Google Colab

This project can be executed using Google Colab, which provides a cloud-based environment with pre-installed machine learning libraries.

### Step 1: Open the Notebook in Colab

Click the button below to open the notebook directly:

https://colab.research.google.com/drive/1vb2nL_HwQx-zJSVjFPok0AUv2aDgMEWc#scrollTo=aad5120c

### Step 2: Mount Google Drive

Since the dataset is downloaded and stored locally, you need to mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
### Step 3:
Store the dataset in Google Drive using this structure:
ML_project/
├── HS/
├── LS/
├── Mix/
├── HS.csv
├── LS.csv
└── Mix.csv

### Step 4

Set the dataset path inside the notebook:

 ```python
DATA_DIR = '/content/drive/MyDrive/ML_project'
```

### Step 5:

Run all cells from top to bottom.

## Google Colab Vs Jupyter Notebook:

This project was implemented using Google Colab due to its cloud-based environment, which provides pre-installed deep learning libraries, GPU support, and seamless integration with Google Drive, Which ensures reproducibility, ease of execution, and efficient handling of audio data, making it more suitable than a local Jupyter environment for this task.

## Installation
```bash
pip install -r requirements.txt
```
## Reproducibility

The project is designed to be fully reproducible:

•Clear dataset structure is provided

•Code runs end-to-end without modification

•All dependencies are listed in requirements.txt




## References

- Goodfellow, I., Bengio, Y., & Courville, A. – *Deep Learning*  
  https://www.deeplearningbook.org/

- Stanford CS231n – Convolutional Neural Networks  
  https://cs231n.github.io/convolutional-networks/

- TensorFlow Documentation  
  https://www.tensorflow.org/

- Librosa Documentation  
  https://librosa.org/doc/latest/index.html

- PhysioNet – Biomedical Signal Database  
  https://physionet.org/

- Towards Data Science – Backpropagation Explained  
  https://towardsdatascience.com/backpropagation-explained-very-simply-2f8e4ed7e8e3

- Understanding Mel Spectrograms  
  https://towardsdatascience.com/mel-spectrograms-explained-5c1c5d5a16f8


## Installation
