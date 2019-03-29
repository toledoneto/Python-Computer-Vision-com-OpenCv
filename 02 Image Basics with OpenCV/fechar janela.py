import cv2

img = cv2.imread('../DATA/00-puppy.jpg')

while True:

    cv2.imshow('Puppy', img)

    milisec = 1

    # basicamente, falamos para o cv2 esperar os milisec passados
    # E
    # se o botão pressionado for o ESC (botão 27 no teclado)
    # quebramos o loop
    # no lugar de 27 (ESC) podemos passar algo como ord('q'), ou seja, direto a letra que queremos
    if cv2.waitKey(milisec) & 0xFF == 27:
        break

cv2.destroyAllWindows()
