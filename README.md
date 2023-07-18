# CNNs vs ANNs with MNIST Dataset
## Overview
This repository compares Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs) using the MNIST dataset, a popular dataset of handwritten digits. The objective is to explore and demonstrate the differences in performance and capabilities between these two types of neural networks in the context of image classification tasks.

## Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0 to 9) and corresponding labels indicating the digit's class. It contains 60,000 training samples and 10,000 test samples.

## CNNs and ANNs
### Convolutional Neural Networks (CNNs)
CNNs are specialized for image-related tasks and are designed to process data with grid-like topology, such as images.
They consist of convolutional layers, pooling layers, and fully connected layers.
Convolutional layers use filters (kernels) to scan across the input and extract local features.
Pooling layers downsample the spatial dimensions, reducing the computational burden and increasing robustness.
CNNs automatically learn hierarchical features, allowing them to capture spatial patterns efficiently.

### Artificial Neural Networks (ANNs)
ANNs are the fundamental building blocks of deep learning. They consist of interconnected nodes organized in layers.
Each node (neuron) in an ANN receives inputs, performs a weighted sum, and applies an activation function to produce an output.
ANNs are fully connected, meaning each node in one layer is connected to every node in the next layer.
They work well for simple tasks but may struggle with complex patterns in images due to their inability to capture spatial relationships.

## Results
We compare the performance of both models based on the following metrics:

Accuracy: The proportion of correct predictions on the test set.

## How to Run
To run the models, follow these steps:

Clone this repository to your local machine.
Install the required dependencies (TensorFlow, Keras, etc.).
Run cnn_mnist.py and ann_mnist.py to train and evaluate the CNN and ANN models, respectively.
Observe and compare the results in terms of accuracy and training time.

## Conclusion
Through this project, we aim to showcase the power and effectiveness of CNNs in image classification tasks, especially when compared to traditional ANNs. The MNIST dataset provides an excellent opportunity to demonstrate the strengths and weaknesses of each approach, helping you gain a better understanding of CNNs and ANNs in the context of computer vision tasks.
