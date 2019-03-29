# câmeras e geral criam ditorções em img, em especial em mov
# geralmente usado para rastrear img em movimento


import cv2
import matplotlib.pyplot as plt


flat_chess = cv2.imread('../DATA/flat_chessboard.png')
plt.imshow(flat_chess)

# esse algoritmo abaixo busca padrões que parecem com um tabuleiro de xadrez e apenas esse tipo de padrão
# found retorna um bool caso seja encontrado padrão
# corners é um array com os pontos dos cantos econtrados
found, corners = cv2.findChessboardCorners(flat_chess, (7, 7))

cv2.drawChessboardCorners(flat_chess, (7, 7), corners, found)
plt.imshow(flat_chess)

# ----------------------------------------------------------
# ------------------------ outra img -----------------------
# ----------------------------------------------------------
dots = cv2.imread('../DATA/dot_grid.png')
plt.imshow(dots)

found, corners = cv2.findCirclesGrid(dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)
cv2.drawChessboardCorners(dots, (10, 10), corners, found)
plt.imshow(dots)

plt.show()
