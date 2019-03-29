# Create a program that reads in a live stream from a camera on your computer
# (or if you don't have a camera, just open up a video file). Then whenever you click the left mouse button down,
# create a blue circle around where you've clicked

# Create a function based on a CV2 Event (Left button click)
import cv2

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global center, clicked

    # get mouse click on down and track center
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        center = (x, y)

    # Use boolean variable to track if the mouse has been released
    if event == cv2.EVENT_LBUTTONUP and clicked:
        clicked = False
        center = (0, 0)


# Haven't drawn anything yet!
center = (0, 0)
clicked = False

# Capture Video
cap = cv2.VideoCapture(0)

# Create a named window for connections
cv2.namedWindow('Test')

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback("Test", draw_circle)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Use if statement to see if clicked is true
    if clicked:
        # Draw circle on frame
        cv2.circle(frame, center, 100, (255, 0, 0), 4)

    # Display the resulting frame
    cv2.imshow('Test', frame)

    # This command let's us quit with the "q" button on a keyboard.
    milisec = 1

    if cv2.waitKey(milisec) & 0xFF == 27:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
