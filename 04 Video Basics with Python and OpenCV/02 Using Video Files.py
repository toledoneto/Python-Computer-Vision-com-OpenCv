import time

import cv2

cap = cv2.VideoCapture('../DATA/finger_move.mp4')

if not cap.isOpened():
    print("Erro! Arqv não encontrado!")

while cap.isOpened():

    ret, frame = cap.read()

    if ret:

        frame_rate = 50

        # precisamos dar um espaço de tempo = ao frame rate para que o vídeo possa ser reproduzido
        time.sleep(1/frame_rate)
        cv2.imshow('frame', frame)

        milisec = 1

        if cv2.waitKey(milisec) & 0xFF == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
