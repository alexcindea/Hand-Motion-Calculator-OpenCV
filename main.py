
import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

class Button:
    def __init__(self, pos, value, height, width):
        self.pos = pos
        self.value = value
        self.height = height
        self.width = width

    def draw(self, image):
        cv2.rectangle(image, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (225, 225, 225), cv2.FILLED)
        cv2.rectangle(image, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (50, 50, 50), 3)
        cv2.putText(image, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)

    def clickCheck(self, image, indexX, indexY):
        if (self.pos[0] < indexX < self.pos[0] + self.width) and (self.pos[1] < indexY < self.pos[1] + self.height):
            cv2.rectangle(image, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (225, 225, 225),cv2.FILLED)
            cv2.rectangle(image, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height), (50, 50, 50), 3)
            cv2.putText(image, self.value, (self.pos[0] + 25, self.pos[1] + 75), cv2.FONT_HERSHEY_PLAIN, 5,(0,0,0), 5)
            return self.value
        else:
            return -1

#webcam capture

cap = cv2.VideoCapture(0);
cap.set(3, 1280)
cap.set(4, 720)

#creating buttons
buttonListValues = [['7', '8', '9', '*'],
                    ['4', '5', '6', '-'],
                    ['1', '2', '3', '+'],
                    ['0', '/', '.', '=']]

buttonList = []
for i in range(4):
    for j in range(4):
        xPos = i * 100 + 800
        yPos = j * 100 + 150
        buttonList.append(Button((xPos, yPos), buttonListValues[j][i], 100, 100))

#using mediapipe hand tracking model
detector = HandDetector(detectionCon = 0.8, maxHands = 1)

#variables
operation = ""
delayCounter = 0
previousButtonPressed = -1

#loop for continuous image proccessing
while True:
    captureSucces, image = cap.read()
    image = cv2.flip(image, 1)
    button_image = np.zeros_like(image)
    hands, image = detector.findHands(image, flipType = False)

    #draw buttons
    cv2.rectangle(button_image, (800, 50), (800 + 400, 70 + 100), (225, 225, 225), cv2.FILLED)
    cv2.rectangle(button_image, (800, 50), (800 + 400, 70 + 100), (50, 50, 50), 3)

    for button in buttonList:
        button.draw(button_image)
    combinedImage = cv2.addWeighted(image, 0.6, button_image, 0.4, 1)

    #processing

    if hands:
        lmList = hands[0]["lmList"]
        length, _, img = detector.findDistance(lmList[8], lmList[12], image)
        x, y, _= lmList[8]
        if length < 40:
            for button in buttonList:

                buttonPressed = button.clickCheck(combinedImage, x, y)
                if buttonPressed != -1 and delayCounter == 0:
                    if(buttonPressed == "="):
                        operation = str(eval(operation))
                    else:
                        if(previousButtonPressed == "="):
                            operation = ""
                        operation += buttonPressed
                    delayCounter = 1
                    previousButtonPressed = buttonPressed

    if delayCounter != 0:
        delayCounter += 1;
        if delayCounter > 10:
            delayCounter = 0


    #displaying the operation + result
    cv2.putText(combinedImage, operation, (800 + 40, 50 + 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 3)

    #display image
    cv2.imshow("webcam feed", combinedImage)
    cv2.waitKey(1)