import cv2

img = cv2.imread('qr.jpg')
result_linear = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LINEAR)
cv2.namedWindow('Linear')
cv2.imshow('Linear', result_linear)

result_nearest = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_NEAREST)
cv2.namedWindow('Nearest')
cv2.imshow('Nearest', result_nearest)

result_area = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_AREA)
cv2.namedWindow('Area')
cv2.imshow('Area', result_area)

result_lanczos = cv2.resize(img, None, fx=2.75, fy=2.75, interpolation=cv2.INTER_LANCZOS4)
cv2.namedWindow('Lanczos4')
cv2.imshow('Lanczos4', result_lanczos)

while True:
    key_code = cv2.waitKey(10)
    if key_code == 27:
        # escape key pressed
        break

cv2.destroyAllWindows()
