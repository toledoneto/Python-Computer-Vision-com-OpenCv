import cv2

# -----------------------------------------------------------
# -----------------------------------------------------------
# ----------------------- DESENHO FIXO ----------------------
# -----------------------------------------------------------
# -----------------------------------------------------------

# cap = cv2.VideoCapture(0)
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # o uso de // garante que o resultado é um int
# # canto sup esq
# y = height // 2
# x = width // 2
#
# # altura e comprimento do retangulo
# w = width // 4
# h = height // 4
#
# # retangulo = x + w , y + h

# -----------------------------------------------------------
# -----------------------------------------------------------
# --------------- DESENHO DINAMICO (CALLBACK) ---------------
# -----------------------------------------------------------
# -----------------------------------------------------------


# FUNÇÃO DE CALLBACK
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, botRight_clicked

    if event == cv2.EVENT_LBUTTONDOWN:

        # caso o rect já tenha sido desenhado, precisamos resetar os valores
        # if topLeft_clicked == True and botRight_clicked == True:
        if topLeft_clicked and botRight_clicked:
            pt1 = (0, 0)
            pt2 = (0, 0)
            topLeft_clicked = False
            botRight_clicked = False

        if not topLeft_clicked:
            pt1 = (x, y)
            topLeft_clicked = True

        elif not botRight_clicked:
            pt2 = (x, y)
            botRight_clicked = True


# VAR GLOBAIS
pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

# CONECTAR O CALLBACK NA FUNÇÃO
cap = cv2.VideoCapture(0)

cv2.namedWindow('Test')
cv2.setMouseCallback("Test", draw_rectangle)

while True:

    ret, frame = cap.read()

    # -----------------------------------------------------------
    # DESENHO FIXO
    # cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=4)
    # cv2.imshow('frame', frame)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # DESENHO DINAMICO (CALLBACK)
    if topLeft_clicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)

    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame, pt1, pt2, color=(0, 0, 255), thickness=3)

    cv2.imshow('Test', frame)

    # -----------------------------------------------------------

    milisec = 1

    if cv2.waitKey(milisec) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
