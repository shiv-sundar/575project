import cv2
import numpy as np
#can I find the skin first, then check what is frequently moving
#
# image = cv2.imread("asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
def nothing(x):
    pass

for i in range(1, 15):
    #allow some time for the camera to do its thing
    ret, frame = cap.read()

edges = cv2.namedWindow('edges')
lower = np.array([0, 40, 0], dtype = "uint8")
upper = np.array([15, 255, 255], dtype = "uint8")
backSub = cv2.createBackgroundSubtractorKNN()

while(ret):
    frame = cv2.flip(frame, 1)
    box = frame[50:410, 760:1080, :]
    fgMask = backSub.apply(box)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fgMask = cv2.erode(fgMask, kernel, iterations = 2)
    fgMask = cv2.dilate(fgMask, kernel, iterations = 2)
    fgMask = cv2.GaussianBlur(fgMask, (17, 17), 0)
    _, contours, _ = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(contours):
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        cv2.drawContours(box, contours, -1, (0, 0, 255), 2)

    skin = cv2.bitwise_and(box, box, mask = fgMask)
    cv2.rectangle(frame, (760, 50), (1080, 410), (0, 255, 0), 2)
    cv2.imshow('original', frame)
    if(cv2.waitKey(1) == ord('q')):
        break
    ret, frame = cap.read()

# plt.subplot(121), plt.imshow(image, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
