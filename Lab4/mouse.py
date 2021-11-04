import cv2
import numpy as np


selected_points = []


def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.circle(image, (x, y), 5, (0, 0, 255))
        selected_points.append([x, y])
    elif event == cv2.EVENT_MBUTTONDOWN:
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 3)


image = cv2.imread('road.jpg')
image = cv2.resize(image, None, fx=0.5, fy=0.5)

cv2.namedWindow('image')
var = {'a': 'b'}
cv2.setMouseCallback('image', on_mouse, param=var)

while True:
    cv2.imshow('image', image)

    if cv2.waitKey(100) == 27:
        break

    if len(selected_points) == 4:
        destination_points = [[0, 0], [200, 0], [200, 900], [0, 900]]
        matrix = cv2.getPerspectiveTransform(
            np.float32(selected_points), np.float32(destination_points))
        result = cv2.warpPerspective(image, matrix, (200, 900))
        cv2.imshow('result', result)

cv2.destroyAllWindows()
