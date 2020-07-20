# This program shows how to build a convolutional neural network
# to classify hand written digits with  Keras API by applying on
# the MNIST digits dataset. 
#
#===============================================================
# Author: Marjan Khamesian
# Date: July 2020
#===============================================================

import numpy as np
import matplotlib.pyplot as plt

# Keras imports for the dataset 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Loading the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualization 
image_index = 56565
print(y_train[image_index]) 
plt.imshow(x_train[image_index], cmap='Greys')

# Check the shape of x_train
print('shape of x_train:', x_train.shape)

# Reshaping the array to 4-dimension to work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalization: dividing the RGB codes to the max RGB value
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model 
model = Sequential()

# Adding the layers
model.add(Conv2D(28, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

# Compiling  
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
# Fitting
model.fit(x=x_train, y=y_train, epochs=20)

# Evaluating the model 
model.evaluate(x_test, y_test)

# Prediction
image_index = 1231
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print('individual prediction:', pred.argmax())
