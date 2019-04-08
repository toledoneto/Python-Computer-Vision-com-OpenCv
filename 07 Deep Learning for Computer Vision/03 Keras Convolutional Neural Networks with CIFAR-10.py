# usando keras e RN Convolucionais para lidar com o dataset CIFAR-10

from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report


# carregando o dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)  # (50000, 32, 32, 3) -> agora temos os 3 canais de cores

# mostrando os dados de uma img
single_img = x_train[0]  # array de 32x32

plt.imshow(single_img, cmap='gray_r')

# convertendo para One Hot encoding: marcar a saída correta como 1
print(y_train.shape)

################################################################
# processando os dados em X

# normalizando manualmente
x_train = x_train/255
x_test = x_test/255

scaled_single = x_train[0]

print(scaled_single.max())  # 1

plt.imshow(scaled_single)

################################################################
# convertendo para One Hot encoding: marcar a saída correta como 1
# (dados, num classes)
# possíveis classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

################################################################
# craindo o modelo
# modelo de camadas sequenciais
model = Sequential()

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D - NUM 1
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu',))

# camada de pooling - NUM 1
model.add(MaxPool2D(pool_size=(2, 2)))

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D - NUM 2
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu',))

# camada de pooling - NUM 2
model.add(MaxPool2D(pool_size=(2, 2)))

# achatando as img
# um array de 1D One Hot Code
model.add(Flatten())

# 128 neuronios na camada densa escondida
model.add(Dense(256, activation='relu'))

# camada final: classificador. Por isso apenas 10 neuronios (um para cada saída)
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

################################################################
# treinando o modelo
model.fit(x_train, y_cat_train, epochs=2)

# vendo o nome do retorno no array de predição
print(model.metrics_names)  # ['loss', 'acc']

# avaliando
model.evaluate(x_test, y_cat_test)

predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

plt.show()
