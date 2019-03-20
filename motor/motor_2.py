import RPi.GPIO as GPIO
import sys
import time

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
        GPIO.output(pin3, False)
        GPIO.output(pin4, True)
        GPIO.output(enB, True)
        time.sleep(2)
        
except KeyboardInterrupt:
    GPIO.cleanup()
    sys.exit()