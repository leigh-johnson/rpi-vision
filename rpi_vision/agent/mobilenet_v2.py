# Python
import logging
import argparse

# App
from rpi_vision.agent.capture import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base

logging.basicConfig()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    args = parser.parse_args()
    return args


def main(args):
    model = MobileNetV2Base(include_top=args.include_top)
    capture_manager = PiCameraStream()
    capture_manager.start()

    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            if args.tflite:
                prediction = model.tflite_predict(frame)
            else:
                prediction = model.predict(frame)
            logging.info(prediction)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
