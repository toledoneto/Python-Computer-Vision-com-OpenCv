import cv2

# 0 é a cam default
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# salvando o video - params:
# nome do arqv
# VideoWriter_fourcc é o codec usado para salvar o video
# qdade de frames/sec
writer = cv2.VideoWriter('myvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

while True:

    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    writer.write(frame)

    cv2.imshow('frame', frame)

    milisec = 1

    if cv2.waitKey(milisec) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
