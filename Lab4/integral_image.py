import cv2
import numpy as np


def integral_image(image: np.ndarray) -> np.ndarray:
    integral = image.astype(np.uint64).copy()

    for i in range(1, integral.shape[1]):
        integral[0, i] = integral[0, i] + integral[0, i - 1]

    for i in range(1, integral.shape[0]):
        integral[i, 0] = integral[i, 0] + integral[i - 1, 0]

    for y in range(1, integral.shape[0]):
        for x in range(1, integral.shape[1]):
            integral[y, x] = integral[y, x] + integral[y - 1, x] + \
                integral[y, x - 1] - integral[y - 1, x - 1]

    return integral


arr = np.array([[4, 5, 2, 1],
                [0, 9, 3, 2],
                [5, 6, 8, 1],
                [2, 3, 0, 0]], dtype=np.uint8)
gallery = cv2.imread('gallery.png', cv2.IMREAD_GRAYSCALE)

arr_integral = integral_image(arr)
gallery_integral = integral_image(gallery)

print(arr)
print(arr_integral)
# cv2.imshow('gallery', gallery)
# cv2.waitKey()
