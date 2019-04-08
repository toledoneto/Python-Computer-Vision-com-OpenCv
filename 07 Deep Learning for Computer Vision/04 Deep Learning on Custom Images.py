import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten


cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)

plt.imshow(cat4)
print(cat4.shape)  # (375, 500, 3)

dog = cv2.imread('../DATA/CATS_DOGS/train/DOG/2.jpg')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)

plt.imshow(dog)
print(dog.shape)  # (199, 188, 3)

#################################################################
# PROBLEMAS
# 1. podemos notar que os shape das img são diferentes, ou seja, teremos que ajustar esse detalhe
# 2. devemos fazer uma RN mais robusta que possa id figuras que estejam olhando para lados diferentes
#   da img de treinamento, ou seja, devemos fazer algumas inversões e alterações nas img de testes
#################################################################

# PROBLEMA 2
# criando img randomicas levemente alteradas
image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'
                               )
plt.imshow(image_gen.random_transform(dog))

################################################################
input_shape = (150, 150, 3)
# craindo o modelo
# modelo de camadas sequenciais
model = Sequential()

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D - NUM 1
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu',))
# camada de pooling - NUM 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D - NUM 1
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu',))
# camada de pooling - NUM 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# camada de convolução usado Conv2D pois estamos lidando com imgs em 2D - NUM 1
model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, activation='relu',))
# camada de pooling - NUM 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# achatando as img
# um array de 1D One Hot Code
model.add(Flatten())

# 128 neuronios na camada densa escondida
model.add(Dense(128))
model.add(Activation('relu'))  # add a func de ativação manualmente

# camada de Dropout - retira alguns neuronios de forma aleatória para evitar overfitting
model.add(Dropout(0.5))

# camada final: classificador. Por isso apenas 1 neuronio (um para cada saída)
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

################################################################
################################################################
################################################################
################################################################
################################################################

# essa predição leva muito tempo, caso queira fazer o seu próprio, descomentar e rodar os cód abaixo.
# Na próx seção haverá a predição com o uso de uma modelo já trienado fornecido no curso

# ################################################################
# # treinando o modelo
# batch_size = 16  # n existe valor correto, depende do trabalho
#
# train_img_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/train',
#                                               target_size=input_shape[:2],
#                                               batch_size=batch_size,
#                                               class_mode='binary'
#                                               )
#
# test_img_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/test',
#                                              target_size=input_shape[:2],
#                                              batch_size=batch_size,
#                                              class_mode='binary'
#                                              )
#
# # nos diz quais são as classes encontradas
# print(train_img_gen.class_indices)  # {'CAT': 0, 'DOG': 1}
#
# # steps_per_epoch é o num de imgs de treino que cada época avaliará
# results = model.fit_generator(train_img_gen, epochs=1, steps_per_epoch=150,
#                               validation_data=test_img_gen, validation_steps=12)
#
# # avaliando
# print(results.history['acc'])
# ################################################################

################################################################
################################################################
################################################################
################################################################
################################################################

################################################################
# usando o modelo preditivo fornecido no curso
new_model = load_model('cat_dog_100epochs.h5')

dog_file = '../DATA/CATS_DOGS/train/Dog/2.jpg'

dog_img = image.load_img(dog_file, target_size=(150, 150))
dog_img = image.img_to_array(dog_img)

print(dog_img.shape)  # (150, 150, 3)

# precisamos entregar no formato (batch_size, x, x, x) para a RN
dog_img = np.expand_dims(dog_img, axis=0)
print(dog_img.shape)  # (1, 150, 150, 3)

dog_img = dog_img/255

# vendo qual a classe que a RN nos dá
print(new_model.predict_classes(dog_img))

# vendo qual a % de segurança que a RN tem na previsão
print(new_model.predict(dog_img))
plt.show()
