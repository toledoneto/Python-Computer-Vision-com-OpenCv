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


# img de moedas
sep_coins = cv2.imread('../DATA/pennies.jpg')
display(sep_coins)

# tentando separar as moedas com processos básicos:
# * media blur: embaçar a img para tirar detalhes como rostos das moedas etc
# * grayscale
# * Binary threshold: tornar preto e branco
# * encontrar os cotornos

# blur
sep_blur = cv2.medianBlur(sep_coins, 25)
display(sep_blur)

# grayscale
gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
display(gray_sep_coins)

# Binary threshold
ret, sep_thresh = cv2.threshold(gray_sep_coins, 127, 255, cv2.THRESH_BINARY_INV)
display(sep_thresh, cmap='gray')

# contornos
contours, hierarchy = cv2.findContours(sep_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    # recebendo apenas o valor que diferencia a posição do contorno
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, 255, 10)

display(sep_coins)

plt.show()
