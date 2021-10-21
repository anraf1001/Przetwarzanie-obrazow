import cv2
import numpy as np

# image = np.array([[1, 2],
#                   [6, 5]], dtype=np.float32)
# # result = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
# # result = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# # result = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# result = cv2.resize(image, (4, 4), interpolation=cv2.INTER_CUBIC)
# print(result)


def empty_callback(value):
    pass


img = cv2.imread('parrot.jpeg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('image')

th = 127
cv2.createTrackbar('Threshold', 'image', th, 255, empty_callback)
cv2.createTrackbar('Type', 'image', 0, 4, empty_callback)

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    th = cv2.getTrackbarPos('Threshold', 'image')
    thresh_type_num = cv2.getTrackbarPos('Type', 'image')
    thresh = cv2.THRESH_BINARY
    if thresh_type_num == 1:
        thresh = cv2.THRESH_BINARY_INV
    elif thresh_type_num == 2:
        thresh = cv2.THRESH_TRUNC
    elif thresh_type_num == 3:
        thresh = cv2.THRESH_TOZERO
    elif thresh_type_num == 4:
        thresh = cv2.THRESH_TOZERO_INV

    ret, img_th = cv2.threshold(img, th, 255, thresh)
    # result = np.zeros_like(img)
    # result[img < th] = 255

    cv2.imshow('image', img_th)

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
