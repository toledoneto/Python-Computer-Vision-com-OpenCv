# passar uma img para tons de branco e preto
# detecção de bordas

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../DATA/rainbow.jpg', 0)

plt.figure()
plt.imshow(img, cmap='gray')

# o que será feito é aplicar um limite: qquer valor acima dele será 255, qquer valor abaixo será 0
# o valor máx permitido é de 255, escolhemos a metade como limite.
# O retorno da func abaixo é PONTO DE CORTE, IMG COM TRESH
plt.figure()
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh1, cmap='gray')

plt.figure()
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh2, cmap='gray')

# esse método permite que seja retornado o valor max passado caso o valor esteja acima do limite
# ou, caso contrário, mantém o valor original
plt.figure()
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
plt.imshow(thresh2, cmap='gray')

# esse método é o oposto do de cima
plt.figure()
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
plt.imshow(thresh2, cmap='gray')

# outra img, palavras cruzadas
img = cv2.imread('../DATA/crossword.jpg', 0)

plt.figure()
plt.imshow(img, cmap='gray')


def show_pic(img):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


show_pic(img)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
show_pic(thresh1)

ret, thresh2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
show_pic(thresh2)

# threshold adaptativos
# O '11' é o nº de pixels na vizinhança que serão usados para calcular a média
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)

blended = cv2.addWeighted(src1=thresh1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)

th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
blended2 = cv2.addWeighted(src1=thresh1, alpha=0.6, src2=th3, beta=0.4, gamma=0)
show_pic(blended2)

plt.show()
