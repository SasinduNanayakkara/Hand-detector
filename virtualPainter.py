import cv2
import numpy as np
import time
import os

import handTrackingModule
import handTrackingModule as htm

########################
brushThickness = 15
eraserThickness = 50
#######################

folderPath = "headers"
myList = os.listdir(folderPath) # define the folder path
overlayList = []
for impath in myList:
    image = cv2.imread(f"{folderPath}/{impath}") # loop through and assign images
    overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)
xp, yp = 0, 0
imageCanvas = np.zeros((720, 1280, 3), np.uint8) # create another canvas

cap = cv2.VideoCapture(0) # turn on web cam
cap.set(3, 1280) # resize the window
cap.set(4, 720)

detector = handTrackingModule.handDetector(detectionCon=0.85) # increase the detection confident level

while True:
    # import the image
    success, img = cap.read()
    img = cv2.flip(img, 1) # flip the webcam video

    # find hand landmarks
    img = detector.findHands(img) # find the hand landmarks
    lmList = detector.findPosition(img, draw=False) # find the position of the hand

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:] # get the index finger x,y values
        x2, y2 = lmList[12][1:] # get the middle finger x, y values

        # check which fingers are up
        fingers = detector.fingersUp() # check the which finger is up and not
        # print(fingers)

        # if selection mode - two fingers are up
        if fingers[0] and fingers[1]: # selection mode
            xp, yp = 0, 0 # set the drawing position
            print("selection Mode")
            # checking for the click
            if y1 < 125: # finger in the menu bar
                if 250 < x1 < 450:  # finger in pink color brush
                    header = overlayList[0] # display pink brush image
                    drawColor = (255, 0, 255) # change color into pink
                elif 550 < x1 < 680: # finger in blue color brush
                    drawColor = (255, 0, 0) # change color into blue
                    header = overlayList[1] # display blue brush image
                elif 740 < x1 < 880: # finger in green brush
                    drawColor = (0, 255, 0) # change color into green
                    header = overlayList[2] # display green brush image
                elif 920 < x1 < 1100: # finger in eraser
                    drawColor = (0, 0, 0) # change color into black
                    header = overlayList[3] # display eraser image
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)  # draw a rectangle between the fingers

        # if drawing mode - index finder is up
        if fingers[0] and fingers[1] == False: # when one finger detected
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED) # draw a circle on tip of the finger
            print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1 # set the finger postions

            if drawColor == (0, 0, 0): # if eraser selected
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness) # draw the line according the finger positions
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness) # draw the eraser
            else: # colors selected
                cv2.line(img, (xp, yp), (x1,y1), drawColor, brushThickness) # draw the lines according to the finger postions
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, brushThickness) # draw the lines in other canvas

            xp, yp = x1, y1 # set the positions
    # advance
    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY) # create gray image
    _, imgInv = cv2.threshold(imgGray,50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_BGR2RGB) # convert the color
    img = cv2.bitwise_and(img, imgInv) # convert black into white
    img = cv2.bitwise_or(img, imageCanvas) # convert colors

    # setting the heading image
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img,0.5, imageCanvas,0.5,0) # display drawings in webcam window with low transparent
    cv2.imshow("Image", img) # display webcam
    cv2.imshow("Canvas", imageCanvas) # display canvas
    cv2.waitKey(1)
