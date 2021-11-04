import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')

image = cv2.imread('parrot.jpeg', cv2.IMREAD_GRAYSCALE)
image = cv2.equalizeHist(image)

# histogram, _ = np.histogram(image, bins=256, range=(0, 256))
# plt.bar(range(0, 256), histogram)
# plt.show()

plt.hist(image.flatten(), bins=256, range=(0, 256))
plt.show()

cv2.imshow('image', image)
cv2.waitKey()

