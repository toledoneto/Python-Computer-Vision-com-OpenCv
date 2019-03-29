# Canny Edge detector - 1986
# Algoritmo multi estágios
# * aplica um filtro gaussiano para suavizar e remover ruído
# * acha os gradientes de intensidade da img
# * aplica uma supressão não máxima para retirar repostas espúrias da detecção de bordas
# * aplica um threshold duplo para determinar bordas potenciais
# * encontra por 'histerese': suprime bordas fracos e não conecta as fortes
# Para img de alta resolução e bordas gerais, é melhor aplicar um blur customizado antes de aplicar esse algoritmo


import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('../DATA/sammy_face.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)  # (375, 486, 3)
plt.figure()
plt.imshow(img)

# recebe as margens da img, porém tbm tem bastante ruído (bordas não interessante)
edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.figure()
plt.imshow(edges)

edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.figure()
plt.imshow(edges)

# valor mediano do pixel
med_val = np.median(img)

lower = int(max(0, 0.7*med_val))
upper = int(min(0, 1.3*med_val))

edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
plt.figure()
plt.imshow(edges)

# aplicando blur antes
blur_img = cv2.blur(img, ksize=(7, 7))
edges = cv2.Canny(image=blur_img, threshold1=lower, threshold2=upper)
plt.figure()
plt.imshow(edges)

plt.show()
