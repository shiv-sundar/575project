import cv2
import numpy as np

image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg")
cv2.imshow('A', image)
cv2.waitKey(0)
