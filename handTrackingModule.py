import cv2
import mediapipe as mp
import time


class handDetector: # main class
    # default constructor
    def __init__(self, mode=False, model_complexity=1, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode # create values
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackingCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        # create hands connections
        self.hands = self.mpHands.Hands(self.mode,  self.maxHands, self.model_complexity,  self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # draw hand connections
        self.tipIds = [4, 8, 12, 16, 20] # set the finger tip points

    def findHands(self, img, draw=True): # detect the hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert the colors
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: # when hand connections exists
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # draw the hand connections
        return img # return the image

    def findPosition(self, img, handNo=0, draw=True): # detect specific position
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # define specific hand
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape # get the positions
                cx, cy = int(lm.x * w), int(lm.y * h) # convert into integer
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy]) # list the positions
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) # draw a circle in specific position
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []

        # thumb
        # if lmList[tipIds[0][1]] < lmList[tipIds[0] - 1][1]: # track the thumb finger showing or not
        #     fingers.append(1)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  # track the other fingers showing or not
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # turn on webcam
    detector = handDetector() # create object
    while True:
        success, img = cap.read() # create image
        img = detector.findHands(img) # draw connection
        lmList = detector.findPosition(img) # draw specific position
        if len(lmList) != 0:
            print(lmList[4]) # print the values of specific position


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # display fps rate

        cv2.imshow("image", img) # display image
        cv2.waitKey(1)


if __name__ == "__main__":
    main() # call the main method
