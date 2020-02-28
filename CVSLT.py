import cv2
import numpy as np
from matplotlib import pyplot as plt

# image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# cv2.imshow('A', image)
# cv2.waitKey(0)
# track = cv2.createTrackbar("newFrame", 'edges', 10, 300, [onChange])
while(ret):
    # edges = cv2.Canny(frame, 125, 175)
    edges = cv2.Canny(frame, 100, 200)
    dim = (int(frame.shape[1]*.6), int(frame.shape[0]*.6))
    edges = cv2.resize(edges, dim)
    frame = cv2.resize(frame, dim)
    edges = cv2.flip(edges, 1)
    frame = cv2.flip(frame, 1)
    cv2.imshow('edges', edges)
    cv2.imshow('orig', frame)
    if(cv2.waitKey(1) == ord('q')):
        break
    ret, frame = cap.read()
# plt.subplot(121), plt.imshow(image, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
