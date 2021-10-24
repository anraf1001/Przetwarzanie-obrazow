import cv2
import numpy as np


def kuwahara(img, window_size=3):
    img_new = img.copy()
    border_size = window_size - 1

    for row in range(border_size, len(img_new) - border_size):
        for col in range(border_size, len(img_new[row]) - border_size):
            mean = np.empty(4)
            std = np.empty(4)

            mean[0], std[0] = cv2.meanStdDev(img[row - border_size: row + 1,
                                             col - border_size: col + 1])
            mean[1], std[1] = cv2.meanStdDev(img[row - border_size: row + 1,
                                             col: col + border_size + 1])
            mean[2], std[2] = cv2.meanStdDev(img[row: row + border_size + 1,
                                             col - border_size: col + 1])
            mean[3], std[3] = cv2.meanStdDev(img[row: row + border_size + 1,
                                             col: col + border_size + 1])

            img_new[row, col] = mean[np.argmax(std)]

    return img_new


img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

img_blurred = kuwahara(img)

cv2.namedWindow('image ori')
cv2.namedWindow('image kuwahara')

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    cv2.imshow('image ori', img)
    cv2.imshow('image kuwahara', img_blurred)

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
