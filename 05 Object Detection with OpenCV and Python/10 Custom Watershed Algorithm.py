# agora vamos fazer um algoritmo customizado
# através de clique de mouse, vamos setar os seeds para o watershed

import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def display(img, cmap='gray', title=''):
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_title(title)


# img de rodovia
road = cv2.imread('../DATA/road_image.jpg')

# img de rodovia copiada
road_copy = np.copy(road)
display(road, title='rodovia')
print(road.shape[:2])  # (600, 800)

marker_image = np.zeros(road.shape[:2], dtype=np.int32)
segments = np.zeros(road.shape, dtype=np.uint8)
print(marker_image.shape)  # (600, 800)
print(segments.shape)  # (600, 800, 3)

# color mapping do matplotlib: usando o tab10 dentre as escolhas possíveis
print(cm.tab10(0))


# --------------------------------------------------------
# cria uma cor do color mapping
def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)


# cria 10 cores usando o tab10
colors = []
for i in range(10):
    colors.append(create_rgb(i))

print(colors)

# --------------------------------------------------------
# ----------------------- algoritmo ----------------------
# --------------------------------------------------------
# escolhas de cores
n_markers = 10  # 0-9
current_marker = 1

# marcadores atualizados pelo watershed
marks_updated = False


# função de callback
def mouse_callback(event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passados para o algoritmo
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)

        # aparecendo na cópia
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)

        marks_updated = True


# loop de funcionamento:
cv2.namedWindow(winname='Road image')
cv2.setMouseCallback('Road image', mouse_callback)

while True:

    cv2.imshow('Watershed segments', segments)
    cv2.imshow('Road image', road_copy)

    k = cv2.waitKey(1)

    if k == 27:
        break

    # limpando as cores com a letra c
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)

    # atalizando as escolhas de cores
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    # atualizando as marcações
    if marks_updated:

        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)

        segments = np.zeros(road.shape, dtype=np.uint8)

        for color_ind in range(n_markers):
            segments[marker_image_copy == color_ind] = colors[color_ind]

        marks_updated = False

cv2.destroyAllWindows()
