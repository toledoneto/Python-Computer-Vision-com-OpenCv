# The Challenge
#
# Your task is to build an image classifier with Keras and Convolutional Neural Networks for the Fashion MNIST dataset.
# This data set includes 10 labels of different clothing types with 28 by 28 grayscale images.
# There is a training set of 60,000 images and 10,000 test images.
#
# Label    Description
# 0        T-shirt/top
# 1        Trouser
# 2        Pullover
# 3        Dress
# 4        Coat
# 5        Sandal
# 6        Shirt
# 7        Sneaker
# 8        Bag
# 9        Ankle boot

import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten


#################################################################
# download the dataset using Keras
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#################################################################
# Use matplotlib to view an image from the data set. It can be any image from the data set.
img = plt.imshow(x_train[0])

#################################################################
# Preprocessing the Data

# Normalize the X train and X test data by dividing by the max value of the image arrays
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

# Reshape the X arrays to include a 4 dimension of the single channel
x_train = x_train.reshape(len(x_train), len(x_train[0]), len(x_train[1]), 1)
x_test = x_test.reshape(len(x_test), len(x_test[0]), len(x_test[1]), 1)

# Convert the y_train and y_test values to be one-hot encoded for categorical analysis by Keras
y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

#################################################################
# Building the Model

# Use Keras to create a model consisting of at least the following layers (but feel free to experiment):
# * 2D Convolutional Layer, filters=32 and kernel_size=(4,4)
# * Pooling Layer where pool_size = (2,2)
# * Flatten Layer
# * Dense Layer (128 Neurons, but feel free to play around with this value), RELU activation
# * Final Dense Layer of 10 Neurons with a softmax activation
# Then compile the model with: loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']

model = Sequential()

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu',))

# camada de pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# achatando as img de 28x28 para 764, formato exigido na última camada, uma vez que ela apenas retornará
# um array de 1D One Hot Code
model.add(Flatten())

# 128 neuronios na camada densa escondida
model.add(Dense(128, activation='relu'))

# camada final: classificador. Por isso apenas 10 neuronios (um para cada saída)
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

#################################################################
# Training the Model

# Train/Fit the model to the x_train set.
model.fit(x_train, y_cat_train, epochs=2)

# salvando o modelo
model.save_weights("assessment.h5")

## usando o modelo preditivo se houver (DESCOMENTAR)
# model = load_model('cat_dog_100epochs.h5')

# vendo o nome do retorno no array de predição
print(model.metrics_names)  # ['loss', 'acc']

# avaliando
print(x_test.shape)  # (10000, 28, 28)
print(y_cat_test.shape)  # (10000, 10)
model.evaluate(x_test, y_cat_test)

predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

plt.show()
