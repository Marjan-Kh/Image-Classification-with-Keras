#### Image-Classification-with-Keras


The goal of this program is to build a Convolutional Neural Network to classify hand written digits with Keras API using the MNIST dataset. 


An accuracy of almost 99% is achieved using the Keras Sequential model with the following structure:

- One Conv2D layer (2D convolution layer) with 28 filter, using the kernel_size=(3,3). 
  Conv2D layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.

- The Conv2D layer is followed by a MaxPooling2D layer with a pool_size=(2,2). 
  Any type of operation can be done in pooling layers. Say, to find the outliers.
  
- Followed by a Flatten layer, flatten 2D arrays to 1D array before building the fully connected layers.

- Followed by a Dense layer with 128 nodes and 'relu' activation.

- Followed by a Dropout layer, which fights with the overfitting by disregarding some of the neurons while training.

- Finally, the output layer is a Dense layer with 10 nodes with 'softmax' activation function for multi-class classification.


