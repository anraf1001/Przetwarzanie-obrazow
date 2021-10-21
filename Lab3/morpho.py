import cv2
import numpy as np


def empty_callback(value):
    pass


img = cv2.imread('parrot.jpeg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('image')
cv2.namedWindow('erosion')
cv2.namedWindow('dilation')
cv2.namedWindow('opening')

_, img_th = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img_th, kernel, iterations = 1)
dilation = cv2.dilate(img_th, kernel, iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.createTrackbar('window', 'image', 1, 10, empty_callback)

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    window_size = 2 * cv2.getTrackbarPos('window', 'image') + 1
    kernel = np.ones((window_size, window_size), np.uint8)
    erosion = cv2.erode(img_th, kernel, iterations = 1)
    dilation = cv2.dilate(img_th, kernel, iterations = 1)
    opening = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)

    cv2.imshow('image', img_th)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
