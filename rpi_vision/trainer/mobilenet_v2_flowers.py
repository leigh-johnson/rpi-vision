# Python
import logging
# Lib
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# App
from rpi_vision.dataset.flowers import FlowerDataset


logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
INPUT_SHAPE = (192, 192, 3)
EPOCHS = 20

if __name__ == '__main__':
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

    model.fit(ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
