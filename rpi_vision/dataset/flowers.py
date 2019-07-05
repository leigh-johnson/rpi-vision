
# Example here:
# https://www.tensorflow.org/beta/tutorials/load_data/images#setup

import tensorflow as tf
import pathlib
import random
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
AUTOTUNE = tf.data.experimental.AUTOTUNE


class FlowerDataset(object):
    def __init__(
        self,
        image_size=(192, 192),
        image_channels=3
    ):
        self.image_size = image_size
        self.image_channels = image_channels

        self.image_label_ds = self.init_dataset()

    def init_dataset(self):
        data_root = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                            fname='flower_photos', untar=True)
        data_root = pathlib.Path(data_root)

        image_paths = list(data_root.glob('*/*'))
        image_paths = [str(path) for path in image_paths]
        random.shuffle(image_paths)
        self.image_paths = image_paths
        label_names = sorted(
            item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((name, index)
                              for index, name in enumerate(label_names))
        self.label_names = label_names

        image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in image_paths]

        return self.init_tensors(image_paths, image_labels)

    def init_tensors(self, image_paths, image_labels):
        ds = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
        image_label_ds = ds.map(self.load_and_preprocess_from_path_label)
        return image_label_ds

    def load_and_preprocess_from_path_label(self, path, label):
        return self.load_and_preprocess_image(path), label

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.image_channels)
        image = tf.image.resize(image, self.image_size)
        # image /= 255.0  # normalize to [0,1] range
        # image = 2*image-1  # normalize to [-1, 1] range
        image = preprocess_input(image)
        return image
