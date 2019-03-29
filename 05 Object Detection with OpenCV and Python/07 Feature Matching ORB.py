# Aprimoramento do Template matching
# não requer cópia exata da img em questão
# usa bordas, cantos, contornos etc para detectar padrões
# usa a distância da img buscada com a img disponível
# Três métodos apresentados:
# * força bruta com ORB Descriptors
# * força bruta com SIFT Descriptors e razão de teste
# * FLANN equivalente

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# img de cx de ceral que procuraremos em imgs maiores
reeses = cv2.imread('../DATA/reeses_puffs.png', 0)
display(reeses)

# img maior onde procuraremos a menor
cereals = cv2.imread('../DATA/many_cereals.jpg', 0)
display(cereals)

# cria o detector
orb = cv2.ORB_create()

# achando os ptos chaves e descritores das imgs
kp1, des1 = orb.detectAndCompute(reeses, None)
kp2, des2 = orb.detectAndCompute(cereals, None)

# criando o obj de detecção por força bruta
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# encontrando as detecções
matches = bf.match(des1, des2)

# ele mostra o quão boa a similaridade é: qto >, pior
single_match = matches[0]
print(single_match.distance)  # 76

len(matches)  # 265

# encontrando as distancias para tds as deteções
matches = sorted(matches, key=lambda x: x.distance)

# desenhando
# como len(matches) = 265 é muito, escolhemos as 25 primeiras semelhanças
reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

display(reeses_matches)

plt.show()
