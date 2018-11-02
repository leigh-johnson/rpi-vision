from detector.capture import PiCameraStream
from detector.models import MobileNetV2Detector


if __name__ == "__main__":
    try:
        detector = MobileNetV2Detector()
        capture_manager = PiCameraStream()
        capture_manager.start()

        while not capture_manager.stopped:
            if capture_manager.frame is not None:
              frame = capture_manager.read()
              prediction = detector.predict(frame)
              print("prediction", prediction)
    except KeyboardInterrupt:
        capture_manager.stop()

