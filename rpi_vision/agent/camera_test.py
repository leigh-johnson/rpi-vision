import time
import picamera
import os

camera = picamera.PiCamera()
try:
    camera.framerate = 60
    camera.resolution = (320, 240)
    camera.start_preview()
    time.sleep(60)
    camera.stop_preview()
finally:
    camera.close()
