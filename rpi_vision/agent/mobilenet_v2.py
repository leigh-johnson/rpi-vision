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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = args.to_dict()
    try:
        model = MobileNetV2Base(**args_dict)
        capture_manager = PiCameraStream()
        capture_manager.start()

        while not capture_manager.stopped:
            if capture_manager.frame is not None:
                frame = capture_manager.read()
                prediction = model.predict(frame)
                logging.info(prediction)
    except KeyboardInterrupt:
        capture_manager.stop()
