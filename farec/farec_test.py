import RPi.GPIO as GPIO  ## button, motor
import sys               ## motor
import time              ## button, motor

import face_recognition  ## face recognition
import cv2               ## camera
import camera            ## camera.get_frame
import os                ## to get get saved img
import numpy as np       ## to calculate difference betweet known-unknowns
from gtts import gTTS    ## speaker
import pygame            ## speaker

import FaceRecog         ## FaceRecognition

import requests



import dlib
from imutils import face_utils
from keras.models import load_model

pygame.mixer.init()
warning = pygame.mixer.Sound("beep1.wav")

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#points for face
JAWLINE_POINTS = list(range(0,17))
RIGHT_EYEBROW_POINTS = list(range(17,22))
LEFT_EYEBROW_POINTS = list(range(22,27))
NOSE_POINTS = list(range(27,36))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
MOUTH_OUTLINE_POINTS = list(range(48,61))
MOUT_INNER_POINTS = list(range(61,68))

model = load_model('models/2018_12_17_22_58_35.h5')
model.summary()


def sms():
    url='https://api-sens.ncloud.com/v1/sms/services/ncp:sms:kr:255911898686:farec_sms/messages'
    headers = {
        "Content-Type":"application/json; charset=utf-8",
        "x-ncp-auth-key":"bEa58nmo89h2vbS7JMij",
        "x-ncp-service-secret":"ee0c430887d34c4b86a6ebaa8a75052e"
    }
    data = {
        "type":"SMS",
        "contentType":"COMM",
        "countryCode":"82",
        "from":"01046434124",
        "to":[
            "01046434124"
        ],
        "content":"emergency"
    }
    res = requests.post(url=url, headers=headers, json=data)


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

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


def drawsiness_detection():
    cap = cv2.VideoCapture(0)
    emergency = False

    lclose = 0
    rclose = 0
    closetime = 0
    closeStart = 0
    closeEnd = 0
    
    notFaceTime = 0
    notFaceStart = 0
    notFaceEnd = 0
    
    while cap.isOpened():
      if(emergency):
        break
      ret, img_ori = cap.read()  #if cant read fram, ret = false. if read fram, ret = true and img_ori = fram
      

      if not ret:  #cant read fram
        break

      img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

      img = img_ori.copy()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert image to grayscale image

      faces = detector(gray) #detect face in fram
      if notFaceTime > 5:
          emergency = True;
      
      #detect face, if face is not detected longer than 1sec warning
      isDetected = len(faces) #if face is detected return 1 else return 0
      if isDetected == 0 and notFaceStart == 0:
          notFaceStart = time.time()
      elif isDetected == 0:
          notFaceTime = time.time() - notFaceStart
      else:
          notFaceStart = 0
          notFaceTime = 0
    
    
      if notFaceTime > 3:
          warning.play()
          
      #if face is detected
      for face in faces:
        #determine the facial landmarks for the face region
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)
        
        #point facial landmarks
        #for idx, point in enumerate(shapes):
            #print(point)
           #pos = (point[0], point[1])
           #cv2.circle(img, pos, 2, color=(0,255,255), thickness=-1)
    
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42]) #left eye points
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48]) #right eye points
    
        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
    
        print(eye_img_l, eye_img_l)
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
    
        if rclose == 1 and lclose == 1 and closetime > 3:   #if close eye longer than 1sec, warning
            lR = lG = lB = rR = rG = rB = 256
            warning.play()
    
        if closetime > 5:
            emergency = True
            break;
        
        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(lB,lG,lR), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(rB,rG,rR), thickness=2)
    
        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (lB,lG,lR), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (rB,rG,rR), 2)
    
      cv2.imshow('result', img)
      if cv2.waitKey(1) == ord('q'):
        break
    
    
def buttonSetting():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def motor(setting):
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
    
    print("Motor Setting OK!")
    
    if setting == "forward":
        
        # forward
        GPIO.output(pin1, True)
        GPIO.output(pin2, False)
        GPIO.output(enA, True)
        GPIO.output(pin3, True)
        GPIO.output(pin4, False)
        GPIO.output(enB, True)
        time.sleep(2)
    elif setting == "stop":
            
        # stop
        GPIO.output(pin1, True)
        GPIO.output(pin2, False)
        GPIO.output(enA, False)
        GPIO.output(pin3, True)
        GPIO.output(pin4, False)
        GPIO.output(enB, False)
        time.sleep(2)
        

def speaker(sentance):
    print('TTS start')
    tts = gTTS(sentance)
    tts.save('obj.mp3')
    print('TTS ok')
                
    print('pygame start')
    pygame.mixer.pre_init(44100, 16, 2 , 4096)
    pygame.init()
    #pygame.mixer.init()
    #obj_out = pygame.mixer.music.load('obj.mp3')
    #pygame.mixer.music.play()
    test = pygame.mixer.Sound("obj.wav")
    test.play()
    pygame.event.wait()
    print('pygame ok')

def faceIdentification():
    #speaker("Start face identification system")
    face_recog = FaceRecog.FaceRecognition()
    print(face_recog.known_face_names)
    name = "Unknown"
    
    for i in range(3):
        print(i)
        name, frame = face_recog.get_frame()
        print("took photo")
        if name != "Unknown":

            greeting = "Hello "+name
            #speaker(greeting)
            break
        
        # take photo interval = 3 sec
        time.sleep(3)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        print(name)
    
        if i == 2:
            #speaker("Get out!")
            print("Get out!")
            cv2.imwrite('stranger.jpg', frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            
            # +++++++++++MESSAGE++++++++++++++
            sms()

    # do a bit of cleanup
    cv2.destroyAllWindows()

def process():
    
    buttonSetting()
    
    buttonCount = 0
    
    buttonCheck1 = False
    buttonCheck2 = False
    
    while True:
        
        input_state = GPIO.input(15)
        # button push count
        if input_state == False:
            buttonCount += 1
            print("Button was pushed!")
            print(buttonCount)
            time.sleep(0.2)
        
        if buttonCount == 1 and buttonCheck1 == False:
            faceIdentification()
            
            buttonCheck1 = True
            print("Face Idendification Done!")
            
        if buttonCount == 2 and buttonCheck2 == False:
            motor("forward")
            # ++drawsiness detection
            drawsiness_detection()
            buttonCheck2 = True
            # ++ motor stop
            motor("stop")
            sms()
            


if __name__ == '__main__':
    process()