import cv2


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 0, 255))
    elif event == cv2.EVENT_MBUTTONDOWN:
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 3)


image = cv2.imread('parrot.jpeg')

cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)

while True:
    cv2.imshow('image', image)

    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()
