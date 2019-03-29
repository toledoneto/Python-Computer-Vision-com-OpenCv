# cantos - junção de dois segmentos onde há uma mudança repentina de brilho
# algoritmo publicado em 1994
# peq mudança no Harris Corner que trouxe melhorias
# * muda a função de score usada como crtério em seu antecessor
# * scoring func: R = min(lambda1, lambda2)

import cv2
import matplotlib.pyplot as plt
import numpy as np


flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

# aplicando Shi-Tomasi
# src,
# nº de cantos desejados (-1 para detectar todos)
corners = cv2.goodFeaturesToTrack(gray_flat_chess, 5, 0.01, 10)

# ele n marca os cantos, então precisamos dar um flat no array de retorno e desenhar os círculos nos msm
corners = np.int0(corners)  # passando de float para int

# achatando e desenhando
for i in corners:
    x, y = i.ravel()  # achatando
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)

plt.imshow(flat_chess)

corners = cv2.goodFeaturesToTrack(gray_flat_chess, 64, 0.01, 10)

for i in corners:
    x, y = i.ravel()  # achatando
    cv2.circle(flat_chess, (x, y), 3, (255, 0, 0), -1)

plt.imshow(flat_chess)

corners = cv2.goodFeaturesToTrack(gray_real_chess, 100, 0.01, 10)

for i in corners:
    x, y = i.ravel()  # achatando
    cv2.circle(real_chess, (x, y), 3, (255, 0, 0), -1)

plt.imshow(real_chess)

plt.show()
