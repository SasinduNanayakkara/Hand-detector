import cv2 # import packages
import mediapipe as mp
import time

# turn on web cam
cap = cv2.VideoCapture(0) # turn on webcam

mpHands = mp.solutions.hands # get hand detection mode
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read() # get the image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the colors
    results = hands.process(imgRGB) # get the image
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape # get the shape positions
                cx, cy = int(lm.x*w), int(lm.y*h) # convert positions into integer
                print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draw the hand connections

    cTime = time.time()
    fps = 1/(cTime - pTime) # set the fps rate
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # display the fps rate

    cv2.imshow("image", img) # display the image
    cv2.waitKey(1)