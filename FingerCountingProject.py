import cv2
import time
import numpy
import os
import handTrackingModule as htm
wCam, hCam = 640, 480


cap = cv2.VideoCapture(0) # turn on web cam
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "fingerImages"
myList = os.listdir(folderPath) # get the images from the file
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}') # get the specific image
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75) # create an object of handDetector module

tipIds = [4, 8, 12, 16, 20] # set the finger tip points

while True:
    success, img = cap.read()
    img = detector.findHands(img) # call the find hands method
    lmList = detector.findPosition(img, draw=False) # find the finger postions
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # thumb
        # if lmList[tipIds[0][1]] < lmList[tipIds[0] - 1][1]: # track the thumb finger showing or not
        #     fingers.append(1)

        for id in range(0,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]: # track the other fingers showing or not
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1) # get the count of showing fingers
        print(totalFingers)

        h, w, c, = overlayList[totalFingers -1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1] # display the relevant finger picture according to the tracking

        cv2.rectangle(img, (20,225), (170, 425), (0,255,0), cv2.FILLED) # draw a rectangele
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0), 25) # draw numbers according to the showing fingers

    cTime = time.time()
    # setup fps rates
    fps = 1/ (cTime - pTime)
    pTime = cTime
    # display fps rate
    cv2.putText(img, f"FPS: {int(fps)}", (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)