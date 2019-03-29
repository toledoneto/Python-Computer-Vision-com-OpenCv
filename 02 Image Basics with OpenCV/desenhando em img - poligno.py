import numpy as np
import cv2
import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

print(blank_img.shape)

plt.figure(1)
plt.imshow(blank_img)

plt.figure(2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text='Hello', org=(10, 500), fontFace=font, fontScale=4,
            color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
plt.imshow(blank_img)

vertices = np.array([[100, 300], [200, 200], [400, 300], [200, 400]], dtype=np.int32)
print(vertices.shape)

# cv2 exige que esteja em 3x3 e, como mostrado acima, está 4x2. Reshape
plt.figure(3)
pts = vertices.reshape((-1, 1, 2))
cv2.polylines(blank_img, [pts], isClosed=True, color=(255, 0, 0))
plt.imshow(blank_img)

plt.show()
