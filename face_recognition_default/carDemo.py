import RPi.GPIO as GPIO  ## button, motor
import sys               ## motor
import time              ## button, motor

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

def process():
    buttonSetting()
    
    buttonCount = 0
    
    while True:
        input_state = GPIO.input(15)
        if input_state == False:
            buttonCount += 1
            print("Button was pushed!")
            print(buttonCount)
            time.sleep(0.2)
        
        if buttonCount == 1:
            recogName = "Unknown"
            #face_recognition
            for i in range(3):
                #recogName, capture = face recog()
                if recogName != "Unknown":
                    print("Hello"+recogName)
                    break
                
            if recogName == "Unknown":
                ## send message
                print("Unknown")
                return
            
            else:
                motorSetting()
            
            print("Face Recognition Done!")
            
        #if buttonCount == 2:
            #drawsiness detection


if __name__ == '__main__':
    process()