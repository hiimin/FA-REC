from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import picamera
from gtts import gTTS
import pygame

import RPi.GPIO as GPIO
import sys

import dlib
import numpy as np
from imutils import face_utils
# from keras.models import load_model
import keyboard


def cameraInput():
    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open("/home/pi/Desktop/pi-face-recognition/ourencodings.pickle", "rb").read())
    detector = cv2.CascadeClassifier("/home/pi/Desktop/pi-face-recognition/haarcascade_frontalface_default.xml")

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()
    count = 0
    start_time = 0

    check = False

    #############################################
    ## -- NEED TO CHANGE VIDEO->CAMERA
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)

        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                if (name != "Unknown"):
                    print("person")
                    print(count)
                    if (count == 0):
                        start_time = time.time()
                        count = count + 1
                    pre_name = name
                    between = (time.time() - start_time) * 1000 / 60
                    if (between >= 2):
                        #########
                        print('TTS start')
                        tts = gTTS(name)
                        tts.save('obj.mp3')
                        print('TTS ok')

                        print('pygame start')
                        pygame.init()
                        pygame.mixer.init()
                        obj_out = pygame.mixer.music.load('obj.mp3')
                        pygame.mixer.music.play()
                        # pygame.event.wait()
                        print('pygame ok')
                        ############
                        check = True
                        break

                        count = 0
                        start_time = 0

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    ##########################################################


def motorSetting():
    # MOTOR 
    GPIO.setmode(GPIO.BCM)
    pin1 = 23
    pin2 = 24
    enA = 18
    pin3 = 7
    pin4 = 8
    enB = 12

    GPIO.setup(enA, GPIO.OUT)
    GPIO.setup(pin1, GPIO.OUT)
    GPIO.setup(pin2, GPIO.OUT)
    GPIO.setup(enB, GPIO.OUT)
    GPIO.setup(pin3, GPIO.OUT)
    GPIO.setup(pin4, GPIO.OUT)

    
    while True:
        print('Motor1')
        GPIO.output(pin1, True)
        GPIO.output(pin2, False)
        GPIO.output(enA, True)
        GPIO.output(pin3, True)
        GPIO.output(pin4, False)
        GPIO.output(enB, True)
        time.sleep(2)

'''
def process():
    while True:
        check = False

        for i in range(3):
            name = cameraInput()

            if name != "unkown":
                check = True
                break

        if check == False:
        ## message

        else:
            ## moter setting
            motorSetting()
            ## drowsiness

'''
if __name__ == '__main__':
    #process()
    cameraInput()