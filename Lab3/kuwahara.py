import cv2
import numpy as np
from timeit import timeit
import kuwahara_rust


def create_img_with_padding(img: np.ndarray, border_size: int):
    img_border_shape = (img.shape[0] + 2 * border_size,
                        img.shape[1] + 2 * border_size)

    img_border = np.zeros(img_border_shape, dtype=np.uint8)

    img_border[border_size: -border_size,
               border_size: -border_size] = img

    # Left
    img_border[border_size: -border_size,
               0: border_size] = \
        img_border[border_size: -border_size,
                   border_size].reshape((len(img), 1))
    # Right
    img_border[border_size: -border_size, -border_size: img_border.shape[1]] = \
        img_border[border_size: -border_size,
                   -border_size - 1].reshape((len(img), 1))
    # Top
    img_border[0: border_size] = img_border[border_size]
    # Bottom
    img_border[-border_size: len(img_border)] = \
        img_border[-border_size - 1]

    return img_border


def kuwahara(img: np.ndarray, window_size: int = 3):
    border_size = window_size - 1
    img_border = create_img_with_padding(img, border_size)
    img_new = img.copy()

    for row in range(img_new.shape[0]):
        for col in range(img_new.shape[1]):
            mean = np.empty(4)
            std = np.empty(4)

            mean[0], std[0] = cv2.meanStdDev(img_border[row: row + window_size,
                                             col: col + window_size])
            mean[1], std[1] = cv2.meanStdDev(img_border[row: row + window_size,
                                             col + border_size: col + 2 * window_size])
            mean[2], std[2] = cv2.meanStdDev(img_border[row + border_size: row + 2 * window_size,
                                             col: col + window_size])
            mean[3], std[3] = cv2.meanStdDev(img_border[row + border_size: row + 2 * window_size,
                                             col + border_size: col + 2 * window_size])

            img_new[row, col] = mean[np.argmin(std)]

    return img_new


# From lab
def apply_kuwahara(image: np.ndarray, window_size: int):
    border_size = window_size // 2
    image_border = create_img_with_padding(image, border_size)
    image_new = np.zeros_like(image)

    for y in range(image_new.shape[0]):
        for x in range(image_new.shape[1]):
            window = image_border[y: y + window_size, x: x + window_size]
            regions = [
                window[0: border_size + 1, 0: border_size + 1],
                window[border_size: window_size, 0: border_size + 1],
                window[0: border_size + 1, border_size: window_size],
                window[border_size: window_size,
                       border_size: window_size]
            ]

            mean_std = [cv2.meanStdDev(r) for r in regions]
            image_new[y,
                      x] = min(mean_std, key=lambda x: x[1])[0]

    return image_new


def benchmark():
    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    t1 = timeit(lambda: kuwahara(img), number=1)
    t2 = timeit(lambda: apply_kuwahara(img, 5), number=1)
    t3 = timeit(lambda: kuwahara_rust.apply_kuwahara(img, 5), number=1)

    print(f'Lab kuwahara: {t2 / t1 * 100:.2f}%')
    print(f'Rust kuwahara: {t3 / t1 * 100:.2f}%')


def main():
    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

    img_blurred = kuwahara(img)
    img_blurred_2 = apply_kuwahara(img, 5)
    img_rust = kuwahara_rust.apply_kuwahara(img, 5)

    cv2.namedWindow('image ori')
    cv2.namedWindow('image kuwahara')
    cv2.namedWindow('image kuwahara 2')
    cv2.namedWindow('image kuwahara rust')

    while True:
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        cv2.imshow('image ori', img)
        cv2.imshow('image kuwahara', img_blurred)
        cv2.imshow('image kuwahara 2', img_blurred_2)
        cv2.imshow('image kuwahara rust', img_rust)

    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # benchmark()
