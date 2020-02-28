import cv2
import numpy as np
from matplotlib import pyplot as plt

# image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# cv2.imshow('A', image)
# cv2.waitKey(0)
def nothing(x):
    pass

edges = cv2.namedWindow('edges')
orig = cv2.namedWindow('orig')
minTrack = cv2.createTrackbar('minval', 'edges', 10, 300, nothing)
maxTrack = cv2.createTrackbar('maxval', 'edges', 11, 300, nothing)
# lower = np.array([72, 97, 130], dtype = "uint8")
# upper = np.array([158, 187, 246], dtype = "uint8")
# min_YCrCb = np.array([0,133,77], np.uint8)
# max_YCrCb = np.array([255,173,127], np.uint8)
lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([13, 255, 255], dtype = "uint8")
while(ret):
    # edges = cv2.Canny(frame, 125, 175)
    min = cv2.getTrackbarPos('minval', 'edges')
    max = cv2.getTrackbarPos('maxval', 'edges')
    t = frame.copy()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame = frame[(int(frame.shape[1]/2) - 100):(int(frame.shape[1]/2) + 100), (int(frame.shape[0]/2) - 100):(int(frame.shape[0]/2) + 100)]
    if(min >= max):
        max = min + 1
        cv2.setTrackbarPos('maxval', 'edges', max)

    skinMask = cv2.inRange(frame, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(t, t, mask = skinMask)
    edges = cv2.Canny(skin, min, max)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
    # draw in blue the contours that were founded
        # cv2.drawContours(output, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
        cv2.rectangle(t,(x,y),(x+w,y+h),(0,255,0),2)
    # for i, c in enumerate(contours):
    #     area = cv2.contourArea(c)
    #     if area > 50:
    #         cv2.drawContours(t, contours, i, (0, 255, 0), 3)

    # print(min)

    # edges = cv2.Canny(frame, min, max)
    dim = (int(frame.shape[1]*.6), int(frame.shape[0]*.6))
    edges = cv2.resize(edges, dim)
    t = cv2.resize(t, dim)
    # frame = cv2.resize(frame, dim)
    skin = cv2.resize(skin, dim)
    edges = cv2.flip(edges, 1)
    t = cv2.flip(t, 1)
    # frame = cv2.flip(frame, 1)
    skin = cv2.flip(skin, 1)
    cv2.imshow('edges', edges)
    cv2.imshow('skin', skin)
    cv2.imshow('orig', t)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # avg_h = np.mean(np.mean(frame[:, :, 0]))
    # avg_s = np.mean(np.mean(frame[:, :, 1]))
    # avg_v = np.mean(np.mean(frame[:, :, 2]))
    if(cv2.waitKey(1) == ord('q')):
        break
    ret, frame = cap.read()

# print("min = " + str(min))
# print("max = " + str(max))
# print("mean hue is: " + str(avg_h))
# print("mean sat is: " + str(avg_s))
# print("mean val is: " + str(avg_v))

# plt.subplot(121), plt.imshow(image, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
