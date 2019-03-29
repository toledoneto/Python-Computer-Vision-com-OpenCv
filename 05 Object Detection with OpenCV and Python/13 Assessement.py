# Object Detection Assessment Project Exercise
# Russian License Plate Blurring

# Your goal will be to use Haar Cascades to blur license plates detected in an image!

import cv2
import matplotlib.pyplot as plt


def display(img, cmap='gray', title=''):
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_title(title)


img = cv2.imread('../DATA/car_plate.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display(img, title='placa orignal')

plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


def detect_plate(img):

    # copiando a img
    plate_img = img.copy()

    # criando o reconhecedor
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=5)

    # início do local da placa
    x_offset = plate_rects[0][0]
    y_offset = plate_rects[0][1]

    # pegando a ROI, ou seja, apenas a placa
    roi = img[y_offset:y_offset+plate_rects[0][3], x_offset:x_offset+plate_rects[0][2]]

    # borrando a placa
    blurred = cv2.blur(roi, ksize=(7, 7))

    # colocando a placa borrada sobre a copia da placa
    plate_img[y_offset:y_offset+plate_rects[0][3], x_offset:x_offset+plate_rects[0][2]] = blurred

    # desenhando o retângulo sobre a placa borrada (largura, altura, width, height)
    for(x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x+w, y+h), (255, 0, 0), 4)

    return plate_img


result = detect_plate(img)
display(result, title='placa borrada')
plt.show()
