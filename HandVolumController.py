import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
##################################
wCam, hCam = 640, 480
##################################

cap = cv2.VideoCapture(0) # turn on webcam
cap.set(3, wCam) # setup webcam window size
cap.set(4, hCam)
Ptime = 0
detector = htm.handDetector() # create an object of the module
# setup volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange() # get the volume range
minVol = volumeRange[0]
maxVol = volumeRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img) # call the find hands method
    lmList = detector.findPosition(img, draw=False) # get the position of the hand
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2] # track the 2nd finger
        x2, y2 = lmList[8][1], lmList[8][2] # track the thumb finger
        cx,cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED) # draw a circle on the thumb finger
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED) # draw a circle on the 2nd finger
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # draw a line between fingers
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # draw a circle center of the line

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range 50 - 300
        # volume range -65 - 0

        vol = np.interp(length, [50,300], [minVol, maxVol]) # covert and match the volume range and finger distance
        volBar = np.interp(length, [50, 300], [400, 150]) # define  volume bar size
        volPer = np.interp(length, [50, 300], [0, 100]) # define volume percentage
        print(vol)
        volume.SetMasterVolumeLevel(vol, None) # setup volume according to the changes

        if length < 50: # if fingers touch each other
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # change the circle color

        cv2.rectangle(img, (50, 150), (85,400), (0,255,0), 3) # draw a volume bar
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED) # change the volume according to the changes
        cv2.putText(img, f" {int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3) # draw the volume percentage

    Ctime = time.time()
    # setup fps rate
    fps = 1/(Ctime - Ptime)
    Ptime = Ctime
    # display fps rate
    cv2.putText(img, f"FPS: {int(fps)}", (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cv2.imshow("Img", img) # display image
    cv2.waitKey(1)