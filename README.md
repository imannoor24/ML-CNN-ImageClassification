# Image Classification with Convolutional Neural Network

This repository contains code for building and training a Convolutional Neural Network (CNN) for image classification. The model is trained on a dataset of natural scenes. 
Dataset has been categorized into six classes: 'buildings', 'forest', 'glacier', 'mountain', 'sea', and 'street'.

## Dataset
The dataset consists of around 25,000 images of size 150x150, distributed across the six categories. 
The training set comprises approximately 14,000 images.
The test set has 3,000 images. 
The prediction set includes 7,000 images.

Dataset Source: [Intel Image Classification Dataset on Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)
- The dataset is also available as a zip archive. To use it:
  - Download [CNNDataset.zip](CNNDataset.zip).


## Requirements
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
- Open and run the Jupyter Notebook: [ImageClassification_CNN.ipynb](ImageClassification_CNN.ipynb)
