import cv2
import matplotlib.pyplot as plt
import numpy as np


# Open the dog_backpack.jpg image from the DATA folder and display it in the notebook...
img = cv2.imread('../DATA/dog_backpack.jpg')
print(type(img))

print(img.shape)
plt.imshow(img)

# ...Make sure to correct for the RGB order
plt.figure(2)
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)

# Flip the image upside down and display it in the notebook
new_img = cv2.flip(fix_img, 0)  # -> eixo horizontal
plt.figure(3)
plt.imshow(new_img)

# Draw an empty RED rectangle around the dogs face and display the image in the notebook
plt.figure(4)
cv2.rectangle(fix_img, pt1=(200, 350), pt2=(600, 730), color=(255, 0, 0), thickness=10)
plt.imshow(fix_img)

#  Draw a BLUE TRIANGLE in the middle of the image
plt.figure(5)
fix_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vertices = np.array([[200, 730], [600, 730], [390, 350]], dtype=np.int32)
pts = vertices.reshape((-1, 1, 2))
cv2.polylines(fix_img2, [pts], isClosed=True, color=(0, 0, 255), thickness=10)
plt.imshow(fix_img2)

# fill in this triangle
plt.figure(6)
fix_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.fillPoly(fix_img3, [pts], color=(0, 0, 255))
plt.imshow(fix_img3)

# plt.show()


# Create a script that opens the picture and allows you to draw empty red circles with RIGHT MOUSE BUTTON DOWN click
def draw_circle(event, x, y, flags, params):

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (0, 0, 255), 10)


cv2.namedWindow(winname='Puppy')

cv2.setMouseCallback('Puppy', draw_circle)

while True:

    cv2.imshow('Puppy', img)

    milisec = 1
    if cv2.waitKey(milisec) & 0xFF == 27:
        break

cv2.destroyAllWindows()
