# Image Classification with Convolutional Neural Network


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Imported Dataset](#imported-dataset)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Accuracy and Loss Graphs](#graphs)

## Introduction
This repository contains code for building and training a Convolutional Neural Network (CNN) for image classification. The model is trained on a dataset of natural scenes. 
Dataset has been categorized into six classes: 'buildings', 'forest', 'glacier', 'mountain', 'sea', and 'street'.

## Dataset
The dataset consists of around 25,000 images of size 150x150, distributed across the six categories. 
The training set comprises approximately 14,000 images.
The test set has 3,000 images. 
The prediction set includes 7,000 images.

Dataset Source: [Intel Image Classification Dataset on Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)


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
- Open and run the Jupyter Notebook: [Intel_ImageClassification.ipynb](Intel_ImageClassification.ipynb)


## Imported Dataset
![image](https://github.com/imannoor24/ML-CNN-ImageClassification/assets/138428244/1420ba01-ef10-40f7-bd8a-7a212ae1fe84)

## Model Architecture
```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 148, 148, 50)      1400      
                                                                 
 max_pooling2d (MaxPooling2  (None, 74, 74, 50)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 50)        22550     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 50)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 100)       45100     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 100)       0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 100)       90100     
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 7, 7, 100)         0         
 g2D)                                                            
                                                                 
 conv2d_4 (Conv2D)           (None, 5, 5, 100)         90100     
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 2, 2, 100)         0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 400)               0         
                                                                 
 dense (Dense)               (None, 100)               40100     
                                                                 
 dense_1 (Dense)             (None, 6)                 606             
=================================================================
Total params: 289956 (1.11 MB)
Trainable params: 289956 (1.11 MB)
Non-trainable params: 0 (0.00 Byte)
```

## Training Results
Epoch 1/25
47/47 [==============================] - 22s 279ms/step - loss: 1.4262 - accuracy: 0.4058 - val_loss: 1.2780 - val_accuracy: 0.4929
Epoch 2/25
...
Epoch 20: early stopping
test loss and accuracy: 0.6466941237449646 0.7835714221000671


## Graphs:
#### Training and Testing Graphs of Accuracy and Loss
![image](https://github.com/imannoor24/ML-CNN-ImageClassification/assets/138428244/51763c33-514e-4746-97db-530214a7f6e4)

