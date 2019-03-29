import numpy as np
import cv2
import matplotlib.pyplot as plt


# Variáveis
drawing = False  # True se o btn estiver pressionado
ix = -1
iy = -1


def draw_rectangle(event, x, y, flags, params):

    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), color=(0, 255, 0), thickness=-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), color=(0, 255, 0), thickness=-1)


cv2.namedWindow(winname='my_drawing')

cv2.setMouseCallback('my_drawing', draw_rectangle)

img = np.zeros((512, 512, 3), np.uint8)

while True:

    cv2.imshow('my_drawing', img)

    millisec = 20

    if cv2.waitKey(millisec) & 0xFF == 27:
        break

cv2.destroyAllWindows()
