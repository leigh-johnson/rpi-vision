# Python
import argparse
import json
# lib
# app
from rpi_vision.trainer.tfhub import TFHubTrainer

EPOCHS = 50
BATCH_SIZE = 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-url',
        default=EPOCHS,
        required=True,
        help='URL of zip/tar with directory stucture label_name/...samples'
    )
    parser.add_argument(
        '--model-url',
        default=EPOCHS,
        required=True,
        help='TFHub model URL'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of iterations (training rounds) to run'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Training sample batch size'
    )
    parser.add_argument(
        '--log-dir',
        required=True,
        help='Directory where Tensorflow checkpoints will be written'
    )
    parser.add_argument(
        '--model-checkpoint-kwargs',
        type=json.loads,
        default="{}",
        help='''
        Default (json):
            {
                "monitor": "val_loss",
                "save_best_only": false,
                "save_weights_only": true,
                "mode": "auto"
            }
        '''
    )
    parser.add_argument(
        '--early-stopping-kwargs',
        type=json.loads,
        default="{}",
        help='''
        Default (json):
            {
                "monitor": "acc",
                "mode": "max"
            }
        '''
    )
    parser.add_argument(
        '--tensorboard-kwargs',
        type=json.loads,
        default="{}",
        help='''
        Default (json):
            {
                "tensorboard_write_images": true,
                "tensorboard_write_graph": true
            }
        '''
    )
    parser.add_argument(
        '--image-shape',
        type=int,
        nargs='+',
        default=(224, 224, 3),
        help='''
        Shape of model input
        Example:
        224 224 3
        (width height channel)
        '''
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)
    trainer = TFHubTrainer(**args_dict)
    trainer.fit()
