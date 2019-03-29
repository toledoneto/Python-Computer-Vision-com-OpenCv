# template matching -> procura a img dentro de outra

import cv2
import matplotlib.pyplot as plt


full = cv2.imread('../DATA/sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
print(full.shape)  # (1367, 1025, 3)
plt.imshow(full)

face = cv2.imread('../DATA/sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
print(face.shape)  # (375, 486, 3)
plt.imshow(face)

# eval é uma função built-in que assimila uma string c uma das funções existentes em python e a transforma
# na função equivalente. P.E:
# mystring = 'sum'
# myfunc = eval(mystring)
# myfunc([1, 2, 3])
# retorna 6

# métodos de template matching
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

full_copy = full.copy()

my_method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(full_copy, face, my_method)

# é basicamente um heatmap com o ponto mais claro sendo o local mais provável de equivalência
# plt.imshow(res)

for m in methods:

    full_copy = full.copy()

    method = eval(m)

    res = cv2.matchTemplate(full_copy, face,  method)

    # achando os valore max e min do heatmap
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # Notice the coloring on the last 2 left hand side images.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = face.shape

    # Assign the Bottom Right of the rectangle
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Draw the Red Rectangle
    cv2.rectangle(full_copy, top_left, bottom_right, 255, 10)

    # Plot the Images
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Result of Template Matching')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detected Point')
    plt.suptitle(m)

    plt.show()

# plt.show()
