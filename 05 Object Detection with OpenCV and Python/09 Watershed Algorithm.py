# Trata a img como curvas de relevo da topografia
# o brilho de cada pto indica sua altura e as linhas indicam o caimento
# img em tom de cinza:
# * alta intensidade: picos
# * baixa intensidade: vales
# algoritmo preenche os vales isolados com labels
# conforme as lables são aproximadas dos picos, podem se confundir com outros vales
# * criado barreiras onde há esse encontro
# podemos informar manualmente onde estão os vales/picos

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# img de moedas usada no processamento
img = cv2.imread('../DATA/pennies.jpg')

# img de moedas usada para avaliação
sep_coins = cv2.imread('../DATA/pennies.jpg')

# blur
img = cv2.medianBlur(img, 35)

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binary threshold
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh, cmap='gray')

# removendo ruídos (opcional)

kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
display(opening)

# descobrindo o que é certamente background (bg)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
display(sure_bg)

# distance transform
# em uma img binária (0 e 1 ou 0 e 255), os valores mais próx da borda 0/1 mantém seu valor
# os valores mais distântes vão sendo somados até que se tenha certa certeza do que é bg e foreground (fg)
# no fim, o centro da img vai tornado-se mais claro e as bordas escurecendo
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
display(dist_transform)

# aplicando um threshold para separar os ptos que são fg com certeza
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
display(sure_fg)

# agora que temos bg e fg, temos que usar watershed para saber a região entre bg e fg
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)
display(unknown, cmap='gray')

# criando os markers para o algoritmo watershed em 3 passos
# criar os markers - setando como sure_fg
ret, markers = cv2.connectedComponents(sure_fg)

# fazendo os markers serem 1
markers = markers+1

# marcando a região desconhecida como zero
markers[unknown == 255] = 0
display(markers, cmap='gray')

markers = cv2.watershed(img, markers)
display(markers)

contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# dúvida: rever cód de contornos
for i in range(len(contours)):

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (255, 0, 0), 10)

display(sep_coins)

plt.show()
