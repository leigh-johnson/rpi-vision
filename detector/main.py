from .capture import PiCameraStream
from .models import MobileNetV2Detector


if __name__ == "__main__":
    try:
        detector = MobileNetV2Detector()
        capture_manager = PiCameraStream()
        capture_manager.start()

        while not capture_manager.stopped:
            capture_manager.flush()
            frame = capture_manager.frame
            prediction = detector.predict(frame)
            print("prediction", prediction)
    except KeyboardInterrupt:
        capture_manager.stop()

