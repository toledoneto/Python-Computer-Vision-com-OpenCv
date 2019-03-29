# cantos - junção de dois segmentos onde há uma mudança repentina de brilho
# algoritmo publicado em 1988
# cantos podem ser detectados olhando em mudanças significantes em tds direções
# scoring func: R = lambda1 * lambda2 -k(lambda1 + lambda2)


import cv2
import matplotlib.pyplot as plt
import numpy as np


flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess, cmap='gray')

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess, cmap='gray')

# aplicando Harris Corner
gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# precisamos dilatar pra ver os resultados
dst = cv2.dilate(dst, None)

# setando como vermelho qquer ponto que seja 1% > que o resultado da dilatação
flat_chess[dst > 0.01*dst.max()] = [255, 0, 0]
plt.imshow(flat_chess, cmap='gray')

gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
real_chess[dst > 0.01*dst.max()] = [255, 0, 0]
plt.imshow(real_chess, cmap='gray')

plt.show()
