# Python
import logging
import argparse
# Lib
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# App
from rpi_vision.dataset.flowers import FlowerDataset


logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 24
INPUT_SHAPE = (192, 192, 3)
EPOCHS = 50
LOG_DIR = 'job-output/flowers'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of iterations (training rounds) to run'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='Restore from latest checkpoint in --job-dir'
    )
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=10,
        help='Maximum number of checkpoint steps to keep'
    )
    parser.add_argument(
        '--job-dir',
        default='./checkpoints',
        help='Directory where Tensorflow checkpoints will be written'
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.restore:
        raise NotImplementedError

    model_base = MobileNetV2(
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    model_base.trainable = False
    dataset = FlowerDataset()

    steps_per_epoch = tf.math.ceil(
        len(dataset.image_paths)/BATCH_SIZE).numpy()

    image_batch, label_batch = next(iter(dataset.image_label_ds))

    model = tf.keras.Sequential([
        model_base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(dataset.label_names))])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ds = dataset.image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=len(dataset.image_paths)))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    logger.info(model.summary())

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, write_images=True, write_graph=False,
            histogram_freq=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            LOG_DIR + '/weights.{epoch:02d}.hdf5', monitor='val_loss',
            save_best_only=False, save_weights_only=False, mode='auto', save_freq=100)
    ]

    model.fit(ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
              callbacks=callbacks
              )
