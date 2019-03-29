# Contornos: curvas que juntam pontos contínuos (pela borda), com a msm cor e intensidade
# úteis para analises em detecção e reconhecimento de objetos


import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('../DATA/internal_external.png', 0)
print(img.shape)  # (652, 1080)
plt.imshow(img, cmap='gray')

# cv2.RETR_CCOMP: traz tds os contornos
# cv2.CHAIN_APPROX_SIMPLE: é o método de detecção de contorno
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# tipos dos dados
print(type(hierarchy))  # np.ndarray do tipo [x, y, z, local]
                        # com: (x, y, z) os pontos do contorno e local dizendo se é interno ou externo
print(hierarchy)
print(hierarchy[0][5][3])
print(type(contours))  # list
print(len(contours))  # 22 o nº de contornos encontrados

# ------------------------------------------------------------------
# apenas os contorno externos
external_contours = np.zeros(img.shape)  # (652, 1080)

for i in range(len(contours)):

    # recebendo apenas o valor que diferencia a posição do contorno
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

plt.imshow(external_contours, cmap='gray')
# ------------------------------------------------------------------
# apenas os contorno internos
external_contours = np.zeros(img.shape)  # (652, 1080)

for i in range(len(contours)):

    # recebendo apenas o valor que diferencia a posição do contorno
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(external_contours, contours, i, 255, -1)

plt.imshow(external_contours, cmap='gray')
# ------------------------------------------------------------------
# apenas os contorno internos DO ROSTO
external_contours = np.zeros(img.shape)  # (652, 1080)

for i in range(len(contours)):

    # recebendo apenas o valor que diferencia a posição do contorno
    if hierarchy[0][i][3] == 0:
        cv2.drawContours(external_contours, contours, i, 255, -1)

plt.imshow(external_contours, cmap='gray')
# ------------------------------------------------------------------
# apenas os contorno internos DA PIZZA
external_contours = np.zeros(img.shape)  # (652, 1080)

for i in range(len(contours)):

    # recebendo apenas o valor que diferencia a posição do contorno
    if hierarchy[0][i][3] == 4:
        cv2.drawContours(external_contours, contours, i, 255, -1)

plt.imshow(external_contours, cmap='gray')

plt.show()
