from rpi_vision.agent.capture import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base


if __name__ == "__main__":
    try:
        detector = MobileNetV2Base()
        capture_manager = PiCameraStream()
        capture_manager.start()

        while not capture_manager.stopped:
            if capture_manager.frame is not None:
                frame = capture_manager.read()
                prediction = detector.predict(frame)
                print("prediction", prediction)
    except KeyboardInterrupt:
        capture_manager.stop()
