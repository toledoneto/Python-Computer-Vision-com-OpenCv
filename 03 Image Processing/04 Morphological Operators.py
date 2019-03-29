# efeitos variados. P.e.:
# * redução de ruído -> reduzir ptos brancos em fundos pretos e vice versa
# * dilatação e erosão de img


import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_img():
    blank_img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50, 300), fontFace=font, fontScale=5, color=(255, 255, 255), thickness=20)
    return blank_img


def display_img(img):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


img = load_img()
display_img(img)

# montando kernel
kernel = np.ones((5, 5), dtype=np.uint8)

# erosão:
result = cv2.erode(img, kernel, iterations=1)
# display_img(result)

result = cv2.erode(img, kernel, iterations=5)
# display_img(result)

# add ruído branco ao fundo entre [0, 2[
white_noise = np.random.randint(low=0, high=2, size=(600, 600))

# colocando o ruído em 255
white_noise = white_noise * 255
# display_img(white_noise)

# add o ruído à img
noise_img = white_noise + img
# display_img(noise_img)

# retirando ruído 1
opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN, kernel)
# display_img(opening)

# add ruído preto ao fundo entre [0, 2[
black_noise = np.random.randint(low=0, high=2, size=(600, 600))

# colocando o ruído em -255
black_noise = black_noise * -255
# display_img(white_noise)

# add o ruído à img
black_noise_img = black_noise + img

# setando os valores em zero
black_noise_img[black_noise_img == -255] = 0
# display_img(black_noise_img)

# retirando ruído 2
closing = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
# display_img(closing)

# gradiente
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)

plt.show()
