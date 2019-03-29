# em geral usado com detecção de bordas. Params:
# * Gamma correction: deixa mais claro ou escuro, dependendo do valor de gamma
# * kernel: uma matrix que pega os NxN pixels mais próx e multiplica por ctes e, por fim, soma os valores e devolve o
#           valor do novo pixel
# Existe o problema com as bordas laterais da img que não possuem os NxN necessários para cálculo


import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_img():
    img = cv2.imread('../DATA/bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def display_img(img):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img)


i = load_img()
# display_img(i)

# gamma claro
gamma = 1/4
result = np.power(i, gamma)
# display_img(result)

# gamma escuro
gamma = 8
result = np.power(i, gamma)
# display_img(result)

# escrita
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='Bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
# display_img(img)

# criando kernel: uma matriz 5x5 de valores = 0.04
kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25

dst = cv2.filter2D(img, -1, kernel)
# display_img(dst)

# outro experimento: resetando a img
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='Bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)

blurred = cv2.blur(img, ksize=(5, 5))
# display_img(blurred)

blurred = cv2.blur(img, ksize=(10, 10))
# display_img(blurred)

# outro experimento: resetando a img
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='Bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)

blurred = cv2.GaussianBlur(img, (5, 5), 10)
# display_img(blurred)

# outro experimento: resetando a img
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='Bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)

# como o kernel é uma matrix quadrada por default, passamos apenas um nº
blurred = cv2.medianBlur(img, 5)
# display_img(blurred)

# outro experimento: resetando a img
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text='Bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)

# é um blur que mantém as letras mais visíveis, embaçando o fundo
blurred = cv2.bilateralFilter(img, 9, 75, 75)
display_img(blurred)

# outra img
img = cv2.imread('../DATA/sammy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display_img(img)

img_noise = cv2.imread('../DATA/sammy_noise.jpg')
img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)
display_img(img_noise)

median = cv2.medianBlur(img_noise, 5)
display_img(median)

plt.show()
