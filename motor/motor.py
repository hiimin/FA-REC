import RPi.GPIO as GPIO
import sys
import time

GPIO.setmode(GPIO.BCM)
pin1 = 23
pin2 = 24
en = 18

GPIO.setup(en, GPIO.OUT)
GPIO.setup(pin1, GPIO.OUT)
GPIO.setup(pin2, GPIO.OUT)

try:
    while True:
        print('Motor1')
        GPIO.output(pin1, True)
        GPIO.output(pin2, False)
        GPIO.output(en, True)
        time.sleep(2)
        
except KeyboardInterrupt:
    GPIO.cleanup()
    sys.exit()