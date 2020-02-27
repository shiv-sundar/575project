import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg")
# cv2.imshow('A', image)
# cv2.waitKey(0)
edges = cv2.Canny(image, 100, 200)
plt.subplot(121), plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
