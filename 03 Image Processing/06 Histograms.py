# histograma é uma representação visual de distruibuição de valores contínuos
# no caso de img, a freq de valores de cores
# podemos colocar o histograma dos 3 canais de cores e ver quanto cada um deles aparece

import cv2
import matplotlib.pyplot as plt
import numpy as np


dark_horse = cv2.imread('../DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

# plt.imshow(show_horse)
# plt.imshow(show_bricks)

# histSize e ranges devem ir até um valor a mais do máx buscado e o canal é o azul
# OpenCv BRG
hist_bricks = cv2.calcHist([blue_bricks], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
hist_horse = cv2.calcHist([dark_horse], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

print(hist_bricks.shape)

# plt.figure()
# plt.plot(hist_horse)
# plt.figure()
# plt.plot(hist_bricks)

plt.figure()
img = blue_bricks

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.title('HISTOGRAMA PARA BLUE BRICKS')

plt.figure()
img = dark_horse

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 50])
    plt.ylim([0, 500000])

plt.title('HISTOGRAMA PARA DARK HORSE')

# -----------------------------------------------------------
# -----------------------------------------------------------
# ------------------ Histograma em uma ROI ------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

img = rainbow
print(img.shape)  # (550, 413, 3)

plt.figure()
mask = np.zeros(img.shape[:2], np.uint8)  # apenas (550, 413)
mask[300: 400, 100: 400] = 255
plt.imshow(mask, cmap='gray')

# rainbow com mask
plt.figure()
masked_img = cv2.bitwise_and(img, img, mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)

plt.imshow(show_masked_img)

# histograma
hist_mask_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
hist_values_red = cv2.calcHist([rainbow], channels=[2], mask=None, histSize=[256], ranges=[0, 256])

plt.figure()
plt.plot(hist_mask_values_red)
plt.title('HISTOGRAMA VERMELHO PARA RAINBOW COM MASK')

plt.figure()
plt.plot(hist_values_red)
plt.title('HISTOGRAMA VERMELHO PARA RAINBOW SEM MASK')

# -----------------------------------------------------------
# -----------------------------------------------------------
# ---------------- Histograma de Equalização ----------------
# -----------------------------------------------------------
# -----------------------------------------------------------
# método de ajuste de contraste baseado no histograma da img

gorilla = cv2.imread('../DATA/gorilla.jpg', 0)


def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)


display_img(gorilla, 'gray')

plt.figure()
hist_values = cv2.calcHist([gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist_values)

# img equalizada
eq_gorilla = cv2.equalizeHist(gorilla)
display_img(eq_gorilla, 'gray')

plt.figure()
hist_eq_values = cv2.calcHist([eq_gorilla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist_eq_values)

color_gorilla = cv2.imread('../DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
display_img(color_gorilla)

# para podermos equalizar uma img colorida em OpenCv, devemos usar o sistema HSV de cores
hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)

# os canais agora são HUE, SATURATION, VALUE, dos quais o último interessa nesse caso
# o canal VALUE é dado por hsv[:,:, 2]
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

display_img(eq_color_gorilla)

plt.show()
