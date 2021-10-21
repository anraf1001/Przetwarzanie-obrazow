import cv2
import numpy as np

def empty_callback(value):
    pass

# img = cv2.imread('lenna_noise.bmp')
img = cv2.imread('lenna_salt_and_pepper.bmp')

imgAvg = cv2.blur(img, (5, 5))
imgGauss = cv2.GaussianBlur(img, (5, 5), 0)
imgMedian = cv2.medianBlur(img, 5)

cv2.namedWindow('original')
cv2.namedWindow('Avg')
cv2.namedWindow('Gauss')
cv2.namedWindow('Median')

avgWin = 5
gaussWin = 5
medianWin = 5
cv2.createTrackbar('Window', 'Avg', avgWin, 10, empty_callback)
cv2.createTrackbar('Window', 'Gauss', gaussWin, 10, empty_callback)
cv2.createTrackbar('std', 'Gauss', 1, 300, empty_callback)
cv2.createTrackbar('Window', 'Median', medianWin, 10, empty_callback)

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    avgWin = cv2.getTrackbarPos('Window', 'Avg')
    newWindowAvg = 2 * avgWin + 1
    imgAvg = cv2.blur(img, (newWindowAvg, newWindowAvg))

    gaussWin = cv2.getTrackbarPos('Window', 'Gauss')
    std = cv2.getTrackbarPos('std', 'Gauss') / 100
    newWindowGauss = 2 * gaussWin + 1
    imgGauss = cv2.GaussianBlur(img, (newWindowGauss, newWindowGauss), std)

    medianWin = cv2.getTrackbarPos('Window', 'Median')
    newWindowMedian = 2 * medianWin + 1
    imgMedian = cv2.medianBlur(img, newWindowMedian)

    cv2.imshow('original', img)
    cv2.imshow('Avg', imgAvg)
    cv2.imshow('Gauss', imgGauss)
    cv2.imshow('Median', imgMedian)

cv2.destroyAllWindows()
