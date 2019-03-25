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
#from keras.models import load_model
import keyboard

MODE = 0

while True:
    try:
        print(MODE)
        MODE = int(input("plz enter number: "))
        if MODE == 0:

            # load the known faces and embeddings along with OpenCV's Haar
            # cascade for face detection
            print("[INFO] loading encodings + face detector...")
            data = pickle.loads(open("/home/pi/Desktop/pi-face-recognition/ourencodings.pickle", "rb").read())
            detector = cv2.CascadeClassifier("/home/pi/Desktop/pi-face-recognition/haarcascade_frontalface_default.xml")

            # initialize the video stream and allow the camera sensor to warm up
            print("[INFO] starting video stream...")
            #vs = VideoStream(src=0).start()
            vs = VideoStream(usePiCamera=True).start()
            time.sleep(2.0)

            # start the FPS counter
            fps = FPS().start()
            count = 0
            start_time = 0

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
                                    
                                    if(name != "Unknown"):
                                        print("person")
                                        print(count)
                                        if(count == 0):
                                            start_time = time.time()
                                            count = count+1
                                        pre_name = name
                                        between = (time.time() - start_time)*1000/60
                                        if(between >= 2):
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
                                            pygame.event.wait()
                                            print('pygame ok')
                                            ############
                            
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
        elif MODE == 1:
            MODE += 1
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

            try:
                while True:
                    print('Motor1')
                    GPIO.output(pin1, True)
                    GPIO.output(pin2, False)
                    GPIO.output(enA, True)
                    GPIO.output(pin3, True)
                    GPIO.output(pin4, False)
                    GPIO.output(enB, True)
                    time.sleep(2)
                    
            except KeyboardInterrupt:
                GPIO.cleanup()
                sys.exit()
            """
            # Sleeping Alarm
            IMG_SIZE = (34, 26)

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

            model = load_model('models/2018_12_17_22_58_35.h5')
            model.summary()

            def crop_eye(img, eye_points):
              x1, y1 = np.amin(eye_points, axis=0)
              x2, y2 = np.amax(eye_points, axis=0)
              cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

              w = (x2 - x1) * 1.2
              h = w * IMG_SIZE[1] / IMG_SIZE[0]

              margin_x, margin_y = w / 2, h / 2

              min_x, min_y = int(cx - margin_x), int(cy - margin_y)
              max_x, max_y = int(cx + margin_x), int(cy + margin_y)

              eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

              eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

              return eye_img, eye_rect

            # main
            #cap = cv2.VideoCapture('videos/1.mp4')
            cap = cv2.VideoCapture(0) #내장 카메라 입력

            lclose = 0
            rclose = 0
            closetime = 0
            closeStart = 0
            closeEnd = 0

            while cap.isOpened():
              ret, img_ori = cap.read()

              if not ret:
                break

              img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

              img = img_ori.copy()
              gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

              faces = detector(gray)

              for face in faces:
                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(shapes)

                eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

                eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                cv2.imshow('l', eye_img_l)
                cv2.imshow('r', eye_img_r)

                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

                pred_l = model.predict(eye_input_l)
                pred_r = model.predict(eye_input_r)

                # visualize
                state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
                state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

                state_l = state_l % pred_l
                state_r = state_r % pred_r

                lB = 0
                rB = 0
                lG = 0
                rG = 0
                lR = 0
                rR = 0

                if pred_l < 0.1:
                    lR = 255
                    lB = 0
                    lG = 0
                    lclose = 1
                else:
                    lB = 255
                    lR = 0
                    lG = 0
                    lclose = 0
                    closetime = 0;

                if pred_r < 0.1:
                    rR = 255
                    rB = 0
                    rG = 0
                    rclose = 1
                else:
                    rB = 255
                    rR = 0
                    rG = 0
                    rclose = 0
                    closetime = 0

                if rclose == 1 and lclose == 1 and closeStart == 0:
                    closeStart = time.time()
                elif rclose == 1 and lclose == 1:
                    closetime = time.time() - closeStart
                else:
                    closeStart = 0

                if rclose == 1 and lclose == 1 and closetime > 1:   #1초 이상 눈을 감았을 경우
                    lR = lG = lB = rR = rG = rB = 256

                cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(lB,lG,lR), thickness=2)
                cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(rB,rG,rR), thickness=2)

                cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (lB,lG,lR), 2)
                cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (rB,rG,rR), 2)

              cv2.imshow('result', img)
              if cv2.waitKey(1) == ord('q'):
                break

            """
        elif MODE == 2:
            break
        
        else:
            continue
    except Exception as e:
        break
        