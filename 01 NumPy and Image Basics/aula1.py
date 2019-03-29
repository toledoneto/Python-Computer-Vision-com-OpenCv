import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# abrindo a img com Pillow
pic = Image.open('../DATA/00-puppy.jpg')

# transformando a img em np.array
plt.figure(1)
pic_arr = np.asarray(pic)
plt.imshow(pic_arr)

# shape da pic: (altura, largura, canais de cores[R, G, B])
print(pic_arr.shape)

# img apenas no canal vermelho
pic_red = pic_arr.copy()

# usando escala de cinza, qto mais branco a img, mais daquela cor haverá
# p.e.: quanto mais branco na escala de vermelho, mais vemelho tem naquela parte da img

#  canal vermelho
plt.figure(2)
plt.imshow(pic_red[:, :, 0], cmap='gray')

#  canal verde
plt.figure(3)
plt.imshow(pic_red[:, :, 1], cmap='gray')

#  canal azul
plt.figure(4)
plt.imshow(pic_red[:, :, 2], cmap='gray')

# img apenas com um canal de cor
plt.figure(5)
pic_red[:, :, 1] = 0  # zerando o verde
pic_red[:, :, 2] = 0  # zerando o azul
plt.imshow(pic_red)   # veremos apenas o vermelho

plt.show()
