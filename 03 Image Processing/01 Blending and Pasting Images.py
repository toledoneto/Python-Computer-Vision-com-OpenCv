# Add uma img a outra em geral é feito pela formula
# new_pixel = alpha * pixel1 + beta * pixel2 + gama

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(img1)

plt.figure(2)
plt.imshow(img2)

print(img1.shape)  # (1401, 934, 3)
print(img2.shape)  # (1280, 1277, 3)

# sobrepondo imgs de tamanhos iguais
# resize
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)
plt.figure(3)
plt.imshow(blended)

# sobrepondo imgs de tamanhos diferentes
img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset: y_end, x_offset: x_end] = small_img
plt.figure(4)
plt.imshow(large_img)

# criando máscaras para sobrepor apenas parte da img
img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

# add manualmente o lugar desejado
x_offset = 934 - 600
y_offset = 1401 - 600

# pegado a região de interesse, ROI, da img de background
roi = img1[y_offset:1401, x_offset:943]  # o canto inferior direito da img

# criando a máscara
plt.figure(5)
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img2gray, cmap='gray')

# fazendo o inverso da img acima
plt.figure(6)
mask_inv = cv2.bitwise_not(img2gray)
print("canais de cores da mask inv: " + str(mask_inv.shape))
plt.imshow(mask_inv, cmap='gray')

# as operações acima retiram os canais de cores da img, precisamos recolocá-los
# add um fundo branco de cores, ou seja, os canis de cores
white_background = np.full(img2.shape, 255, dtype=np.uint8)
print("canais de cores da mask inv com fundo branco: " + str(white_background.shape))

# pegando as img e aplicando a mask
plt.figure(7)
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
print("canais de cores do bk: " + str(bk.shape))
plt.imshow(bk)

# pegando a img da parte da frente
plt.figure(8)
fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
print("canais de cores da img2 com mask é: " + str(fg.shape))
plt.imshow(fg)

# aplicando no ROI
plt.figure(9)
final_roi = cv2.bitwise_or(roi, fg)
plt.imshow(final_roi)

# aplicando mask na img total
plt.figure(10)
large_img = img1
small_img = final_roi

# aplicando o ROI com mask sobre a img antiga
large_img[y_offset: y_offset+small_img.shape[0], x_offset: x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)

plt.show()
