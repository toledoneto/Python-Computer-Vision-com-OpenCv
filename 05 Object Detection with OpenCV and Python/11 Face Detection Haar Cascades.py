# detecção de face com Haar Cascade
# detecção de rostos com base em alguns atributos da face
#

import cv2
import matplotlib.pyplot as plt


def display(img, cmap='gray', title=''):
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_title(title)


nadia = cv2.imread('../DATA/Nadia_Murad.jpg', 0)
denis = cv2.imread('../DATA/Denis_Mukwege.jpg', 0)
solvay = cv2.imread('../DATA/solvay_conference.jpg', 0)

display(nadia, title='nadia')
display(denis, title='denis')
display(solvay, title='solvay')

# algoritmo
face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')


def detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for(x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return face_img


result_denis = detect_face(denis)
result_nadia = detect_face(nadia)
result_solvay = detect_face(solvay)

display(result_nadia, title='nadia')
display(result_denis, title='denis')
display(result_solvay, title='solvay')


# algoritmo melhorado
def adj_detect_face(img):

    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for(x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)

    return face_img


result_solvay = adj_detect_face(solvay)
display(result_solvay, title='solvay')

# algoritmo para olhos
eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')


def detect_eyes(img):
    face_img = img.copy()

    eyes = eye_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img


result_nadia = detect_eyes(nadia)
display(result_nadia, title='nadia')


