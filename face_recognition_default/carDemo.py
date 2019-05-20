#import RPi.GPIO as GPIO  ## button, motor
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

'''
# Button setting
def buttonSetting():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def motorSetting():
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
'''
def speaker(sentance):
    print('TTS start')
    tts = gTTS(sentance)
    tts.save('obj.mp3')
    print('TTS ok')
                
    print('pygame start')
    pygame.init()
    pygame.mixer.init()
    obj_out = pygame.mixer.music.load('obj.mp3')
    pygame.mixer.music.play()
    pygame.event.wait()
    print('pygame ok')

def faceIdentification():
    speaker("Start face identification system")
    face_recog = FaceRecog.FaceRecognition()
    print(face_recog.known_face_names)
    name = "Unknown"
    
    for i in range(3):
        print(i)
        name, frame = face_recog.get_frame()
        print("took photo")
        if name != "Unknown":

            greeting = "Hello "+name
            speaker(greeting)
            break
        
        # take photo interval = 3 sec
        time.sleep(3)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        print(name)
    
        if i == 2:
            speaker("Get out!")
            print("Get out!")
            cv2.imwrite('stranger.jpg', frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            
            # +++++++++++MESSAGE++++++++++++++

    # do a bit of cleanup
    cv2.destroyAllWindows()

def process():
    '''
    buttonSetting()
    '''
    buttonCount = 0
    
    while True:
        '''
        input_state = GPIO.input(15)
        # button push count
        if input_state == False:
            buttonCount += 1
            print("Button was pushed!")
            print(buttonCount)
            time.sleep(0.2)
        '''
        if buttonCount == 0:
            faceIdentification()
            
            print("Face Idendification Done!")
            buttonCount += 1
            
        #if buttonCount == 2:
            # ++drawsiness detection


if __name__ == '__main__':
    process()