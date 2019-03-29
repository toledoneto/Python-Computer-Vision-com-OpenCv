# Optical Flow
# padrão de aparente movimento entre dois frames consecutivos causados pelo mov do obj ou da cam
# Assumimos:
# * a intensidade do pixel n varia entre frames consecutivos
# * pixels vizinhos têm mov similar
# OpenCv
# * pega um cnj de pixels de um frame e tenta achá-los no próx frame
# * usa func Lucas-Kanade
# * precisamos informar ao OpenCv quais ptos queremos seguir

# Lucas-Kanade segue apenas os pixels que mandamos seguir
# Caso seja interessante notar quaisquer mudanças, usamos Gunner Farneback para calcular o Dense Optical Flow
# * tudo começa em preto, onde houver movimento pintamos de outra cor

import numpy as np
import cv2

# maxCorners: n de cantos que vamos seguir
corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)

# maxLevel é o nome do posição da pirâmide de img: cada lv é 50% da img no lv anterior
# no lvl 2, temos 1/4 da img original
lk_params = dict(winSize=(200, 200),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# pontos para seguir
prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)

    # status retorna 1 se o mov foi detectado
    good_new = nextPts[status == 1]
    good_prev = prevPts[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        # achata os vetores de entrada
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()

        # cria a linha que marcará os pontos sendo seguidos
        mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)

        # cria o círculo que marcará os pontos a serem seguidos
        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    # fazendo o reshape para funcionar na realimentação da função LK
    prevPts = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
