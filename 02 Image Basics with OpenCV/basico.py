import cv2
import matplotlib.pyplot as plt


img = cv2.imread('../DATA/00-puppy.jpg')
print(type(img))

print(img.shape)

# cv2 e matplotlib esperam ordem diferentes dos canais RGB
# matplotlib -> RGB
# cv2 -> BGR
plt.figure(1)
plt.imshow(img)

# arrumando a img para cv2
plt.figure(2)
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)

# em escala de cinza
plt.figure(3)
img_gray = cv2.imread('../DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
# veremos que existe apenas um canal com valores variando de 0 a 255
print(img_gray.shape)
# porém a img mostrada n estará em cinza devido ao color map default do cv2...
plt.imshow(img_gray)
# para alterar isso, mudamos o cmap
plt.figure(4)
plt.imshow(img_gray, cmap='gray')

# fazendo reshape na img
print(fix_img.shape)
new_img = cv2.resize(fix_img, (1000, 400))  # (largura, altura)
plt.figure(5)
plt.imshow(new_img)

# fazendo reshape com razões de mudança
h_ratio = 0.5
w_ratio = 0.5
new_img = cv2.resize(fix_img, (0, 0), fix_img, w_ratio, h_ratio)
print(new_img.shape)
plt.figure(6)
plt.imshow(new_img)

# rotacionando a img nos eixos
new_img = cv2.flip(fix_img, 0)  # -> eixo horizontal
plt.figure(7)
plt.imshow(new_img)

new_img = cv2.flip(fix_img, 1)  # -> eixo vertical
plt.figure(8)
plt.imshow(new_img)

new_img = cv2.flip(fix_img, -1)  # -> ambos os eixos
plt.figure(9)
plt.imshow(new_img)

# salvando uma img
cv2.imwrite('new_puppy_img.jpg', new_img)

# reduzindo o tamanho da img
fig = plt.figure(figsize=(10, 8))  # nova figura
ax = fig.add_subplot(111)  # criando um subplot de uma única figura
ax.imshow(fix_img)

plt.show()
