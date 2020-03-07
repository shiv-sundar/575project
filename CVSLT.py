import cv2
import numpy as np
from matplotlib import pyplot as plt
#can I find the skin first, then check what is frequently moving
#
# image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
def nothing(x):
    pass

edges = cv2.namedWindow('edges')
orig = cv2.namedWindow('orig')
# minTrack = cv2.createTrackbar('minval', 'edges', 100, 300, nothing)
# maxTrack = cv2.createTrackbar('maxval', 'edges', 111, 300, nothing)
# lower = np.array([72, 97, 130], dtype = "uint8")
# upper = np.array([158, 187, 246], dtype = "uint8")
lower = np.array([0, 30, 0], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")
# for i in range(1, 30):
#     #allow some time for the camera to do its thing
#     ret, frame = cap.read()

# backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()
# aWeight = .5
#
# for i in range(1, 40):
#     bg = frame.copy().astype("float")
#     cv2.accumulateWeighted(frame, bg, aWeight)
#     ret, frame = cap.read()

while(ret):
    frame = cv2.flip(frame, 1)
    fgMask = backSub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    fgMask = cv2.dilate(fgMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    fgMask = cv2.GaussianBlur(fgMask, (15, 15), 0)
    skin = cv2.bitwise_and(frame, frame, mask = fgMask)
    # diff = cv2.absdiff(bg.astype("uint8"), frame)
    #
    # # threshold the diff image so that we get the foreground
    # thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    #
    # # get the contours in the thresholded image
    # _, cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # return None, if no contours detected
    # if len(cnts) != 0:
    #     # based on contour area, get the maximum contour which is the hand
    #     segmented = max(cnts, key=cv2.contourArea)
    # # edges = cv2.Canny(frame, 125, 175)
    # hand = (thresholded, segmented)
    #
    # # check whether hand region is segmented
    # if hand is not None:
    #     # draw the segmented region and display the frame
    #     # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    #     cv2.imshow("Thesholded", thresholded)


    min = cv2.getTrackbarPos('minval', 'edges')
    max = cv2.getTrackbarPos('maxval', 'edges')
    t = frame.copy()
    t = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame = frame[(int(frame.shape[1]/2) - 100):(int(frame.shape[1]/2) + 100), (int(frame.shape[0]/2) - 100):(int(frame.shape[0]/2) + 100)]
    if(min >= max):
        max = min + 1
        cv2.setTrackbarPos('maxval', 'edges', max)
    #
    skinMask = cv2.inRange(t, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 3)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 3)
    blur the mask to help remove noise, then apply the
    mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (15, 15), 0)
    skin = cv2.bitwise_and(frame, skin, mask = skinMask)
    skinMask = cv2.erode(skinMask, kernel, iterations = 3)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 3)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (15, 15), 0)
    edges = cv2.Canny(skin, 100, 200)
    _, contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # if len(contours) != 0:
    # # draw in blue the contours that were founded
    #     # cv2.drawContours(output, contours, -1, 255, 3)
    #
    # # find the biggest countour (c) by the area
    #     # cnts = imutils.grab_contours(cnts)
    #     # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # #     c = max(contours, key = cv2.contourArea)
    # #     x,y,w,h = cv2.boundingRect(c)
    # #
    # # # draw the biggest contour (c) in green
    # #     cv2.rectangle(t,(x,y),(x+w,y+h),(0,255,0),2)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 50:
            cv2.drawContours(skin, contours, i, (0, 255, 0), 3)
    #
    # # print(min)
    #
    # # edges = cv2.Canny(frame, min, max)
    dim = (int(frame.shape[1]*.5), int(frame.shape[0]*.5))
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
    # cv2.imshow('mask', skin)
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
