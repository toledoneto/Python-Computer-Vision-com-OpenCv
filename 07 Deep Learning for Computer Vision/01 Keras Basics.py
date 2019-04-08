# vamos criar uma IA para identificar se a nota é falsa ou verdadeira

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


# dataset em forma de array para formar a img. Na última posição do array, temos um label
# * 0 = nota falsa
# * 1 = nota autentica
data = genfromtxt('../DATA/bank_note_data.txt', delimiter=',')

print(type(data))  # array

# colocando tds as labels em um unico array
labels = data[:, 4]
print(labels)  # [0. 0. 0. ... 1. 1. 1.]

# pegando tudo que são features
features = data[:, 0:4]

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# fazendo o escalonamento dos valores
scaler = MinMaxScaler()

# esse obj acha o min e o max e transforma tds os dados em relação a esses dois valores
# fazemos o fit apenas nos dados de treino para que não seja influenciado pelos dados de teste
scaler.fit(X_train)

# a versão escalonada está aqui
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

############################################################
# trabalhando com Keras

# criando no modelo de RN
model = Sequential()

# RN de 3 camadas
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(scaled_X_train, y_train, epochs=50, verbose=True)

# predizendo test set
predictions = model.predict_classes(scaled_X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# # salvando o modelo
# model.save("my_model.h5")
#
# # carregando o modelo salvo
# from keras.models import load_model
# model = load_model("my_model.h5")
