# Fashion MNIST Classification using CNN in Keras

## Introduction

This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset using Keras and TensorFlow. The Fashion MNIST dataset consists of 70,000 grayscale images of 10 different fashion categories, each of size 28x28 pixels.

## Dataset

The Fashion MNIST dataset is available in Keras and includes:

- 60,000 training images
- 10,000 test images
- 10 classes:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## Project Workflow

1. Load and preprocess the Fashion MNIST dataset.
2. Build a CNN model using Keras.
3. Train the CNN model on the dataset.
4. Evaluate the model on test data.
5. Visualize model performance using accuracy and loss plots.

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn

Install dependencies using:

```sh
pip install tensorflow keras numpy matplotlib seaborn
```

## Model Architecture

The CNN model consists of:

- Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Fully connected layers with dropout
- Softmax activation for classification

## Usage

Run the following script to train and evaluate the model:

```sh
python fashion_mnist_cnn.py
```

## Results

The model achieves approximately 90% accuracy on the test set. Performance can be improved using data augmentation and hyperparameter tuning.

## Results



The model achieves approximately 90% accuracy on the test set. Performance can be improved using data augmentation and hyperparameter tuning.Future Enhancements

- Implement data augmentation to improve generalization.
- Experiment with different architectures like ResNet and MobileNet.
- Fine-tune a pre-trained model on Fashion MNIST.

## Author

[Bikash Konwar]



