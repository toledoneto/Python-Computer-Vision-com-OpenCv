# Dense Optical Flow

# Lucas-Kanade segue apenas os pixels que mandamos seguir
# Caso seja interessante notar quaisquer mudanças, usamos Gunner Farneback para calcular o Dense Optical Flow
# * tudo começa em preto, onde houver movimento pintamos de outra cor

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

# frame antigo
ret, frame1 = cap.read()

# tom de cinza
prvsImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# mask HSV totalmente saturada no canal VALUES
hsv_mask = np.zeros_like(frame1)
hsv_mask[:, :, 1] = 255

while True:
    # próx frame
    ret, frame2 = cap.read()
    nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # params no final do cód
    flow = cv2.calcOpticalFlowFarneback(prvsImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # colorindo baseado no angulo de movimento
    # a func cv2.calcOpticalFlowFarneback nos retorna um vetor cartesiano de direção
    # então vamos convertê-lo para uma notação polar de valores
    # flow[:, :, 0] = info horizontal
    # flow[:, :, 1] = info vertical
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    # o angulo vai determinar a cor do movimento em cada direção
    hsv_mask[:, :, 0] = ang / 2
    # a magnitude vai determinar a saturação
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convertendo pra BGR para aparecer no vídeo
    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # organizando a entrada do loop
    prvsImg = nextImg

cap.release()
cv2.destroyAllWindows()

# params da função
# * prev first 8-bit single-channel input image.
# * next second input image of the same size and the same type as prev.
# * flow computed flow image that has the same size as prev and type CV_32FC2.
# * pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image
#     * pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
# * levels number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
# * winsize averaging window size
#     * larger values increase the algorithm robustness to image
# * noise and give more chances for fast motion detection, but yield more blurred motion field.
# * iterations number of iterations the algorithm does at each pyramid level.
# * poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel
#     * larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
# * poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
