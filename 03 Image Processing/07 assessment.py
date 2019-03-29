import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)


# Open and display the giraffes.jpg
img = cv2.imread('../DATA/giraffes.jpg')
img_color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img original
display_img(img)
plt.title("IMG Original")

# img gray
img_gray = cv2.imread('../DATA/giraffes.jpg', 0)
display_img(img_gray, 'gray')
plt.title("IMG Gray")

# img RGB
display_img(img_color_rgb)
plt.title("IMG RGB")

# Apply a binary threshold onto the image
img_gray = cv2.imread('../DATA/giraffes.jpg', 0)
ret, th_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
display_img(th_binary, cmap='gray')
plt.title("IMG binary threshold")

# convert its colorspace to HSV
img = cv2.imread('../DATA/giraffes.jpg')
img_color_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
display_img(img_color_hsv)
plt.title("IMG HSV")

# Create a low pass filter with a 4 by 4 Kernel filled with values of 1/10 and use 2-D Convolution to blur the image
kernel = np.ones((4, 4), dtype=np.uint8)
kernel = kernel * 0.1

dst = cv2.filter2D(img_color_rgb, -1, kernel)
display_img(dst)
plt.title("IMG 2D Convolution Blur")

#  Create a Horizontal Sobel Filter with a kernel size of 5 to the grayscale version
#  of the giaraffes image and then display the resulting gradient filtered version of the image
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
display_img(sobelx, 'gray')

# Plot the color histograms for the RED, BLUE, and GREEN channel of the giaraffe image
plt.figure()
color = ('r', 'g', 'b')

for i, col in enumerate(color):
    histr = cv2.calcHist([img_color_rgb], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 300])
    plt.ylim([0, 50000])

plt.title('HISTOGRAMA PARA GIRAFFES')

plt.show()
