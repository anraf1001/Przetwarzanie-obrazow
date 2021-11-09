import cv2
import numpy as np
from timeit import timeit


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


def sum_from_integral_img(row_start: int,
                          row_stop: int,
                          column_start: int,
                          column_stop: int,
                          integral_img: np.ndarray) -> int:
    A, B, C = 0, 0, 0

    if row_start != 0:
        B = integral_img[row_start - 1, column_stop]

        if column_start != 0:
            A = integral_img[row_start - 1, column_start - 1]
            C = integral_img[row_stop, column_start - 1]

    elif column_start != 0:
        C = integral_img[row_stop, column_start - 1]

    D = integral_img[row_stop, column_stop]

    return int(D - C - B + A)


arr = np.array([[4, 5, 2, 1],
                [0, 9, 3, 2],
                [5, 6, 8, 1],
                [2, 3, 0, 0]], dtype=np.uint8)
gallery = cv2.imread('gallery.png', cv2.IMREAD_GRAYSCALE)

arr_integral = integral_image(arr)
gallery_integral = integral_image(gallery)

print(arr)
print(arr_integral)

t1 = timeit(lambda: np.sum(arr[1:, 1:]), number=100)
t2 = timeit(lambda: sum_from_integral_img(
    1, 3, 1, 3, arr_integral), number=100)
print(f'Region sum using numpy: {np.sum(arr)}')
print(
    f'Region sum using integral image: {sum_from_integral_img(0, 3, 0, 3, arr_integral)}')
print(f't2/t1 * 100%: {t2/t1 * 100:.2f}%')
# cv2.imshow('gallery', gallery)
# cv2.waitKey()
