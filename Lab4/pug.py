import cv2
import numpy as np

gallery = cv2.imread('gallery.png')
pug = cv2.imread('pug.png')

source_points = np.float32([[0, 0],
                            [pug.shape[1], 0],
                            [pug.shape[1], pug.shape[0]],
                            [0, pug.shape[0]]])
destination_points = np.float32([[598, 360],
                                 [681, 350],
                                 [683, 453],
                                 [598, 442]])

matrix = cv2.getPerspectiveTransform(source_points, destination_points)
pug_result = cv2.warpPerspective(
    pug, matrix, (gallery.shape[1], gallery.shape[0]))

_, pug_th = cv2.threshold(pug_result, 0, 255, cv2.THRESH_BINARY)
pug_closed = cv2.morphologyEx(pug_th,
                              cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))

gallery[pug_closed != 0] = 0
gallery += pug_result

cv2.imshow('gallery', gallery)
cv2.waitKey()
