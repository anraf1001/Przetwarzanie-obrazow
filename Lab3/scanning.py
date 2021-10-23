import cv2

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img.reshape(img.shape[0] * img.shape[1])[::3] = 255

cv2.namedWindow('image')

while True:
    # sleep for 10 ms waiting for user to press some key, return -1 on timeout
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

    cv2.imshow('image', img)

# closes all windows (usually optional as the script ends anyway)
cv2.destroyAllWindows()
