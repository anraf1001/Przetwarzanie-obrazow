import cv2
import numpy as np
from timeit import timeit


def blur(img):
    img_new = img.copy()
    for row in range(1, len(img) - 1):
        for col in range(1, len(img[row]) - 1):
            img_new[row, col] = np.mean(img[row-1:row+2, col-1:col+2])

    return img_new


def builtin_blur(img):
    return cv2.blur(img, (3, 3))


def filter2D_blur(img):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, -1, kernel)


img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img.reshape(img.shape[0] * img.shape[1])[::3] = 255

t1 = timeit(lambda: blur(img), number=1)
t2 = timeit(lambda: builtin_blur(img), number=1)
t3 = timeit(lambda: builtin_blur(img), number=1)
print(t1 / t2)
print(t1 / t3)

img_blurred = blur(img)
img_builtin = builtin_blur(img)
img_filter2D = filter2D_blur(img)

cv2.namedWindow('image')
cv2.namedWindow('image built in blur')
cv2.namedWindow('image filter 2D')

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    cv2.imshow('image', img_blurred)
    cv2.imshow('image built in blur', img_builtin)
    cv2.imshow('image filter 2D', img_filter2D)

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
