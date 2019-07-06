# Python
import logging
import argparse

# App
from rpi_vision.agent.capture import PiCameraStream
from rpi_vision.models.tfhub import TFHubModel

logging.basicConfig()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    args = parser.parse_args()
    return args


def main(args):
    model = TFHubModel()
    capture_manager = PiCameraStream()
    capture_manager.start()

    try:
        while not capture_manager.stopped:
            if capture_manager.frame is not None:
                frame = capture_manager.read()
                prediction = model.predict(frame)
                logging.info(prediction)
    except KeyboardInterrupt:
        capture_manager.stop()


if __name__ == "__main__":
    args = parse_args()
    main(args)
