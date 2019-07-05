# Python
import logging
import os
from time import time
# lib
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
# app


class TFHubModelLayer(object):
    '''
        Google colab examples: https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
    '''

    def __init__(
        self,
        input_shape=(224, 224, 3),
        model_url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4',
        labels_url='https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt',
        trainable=False
    ):

        self.model_url = model_url
        self.input_shape = input_shape
        self.labels_url = labels_url

        self.model = hub.KerasLayer(
            model_url, input_shape=input_shape, trainable=trainable)

        labels_filename = labels_url.split(
            '/')[-1]
        labels_path = tf.keras.utils.get_file(labels_filename, labels_url)
        self.labels = np.array(open(labels_path).read().splitlines())

    def predict(self):
        pass


class TFHubTrainer(object):

    default_model_checkpoint_kwargs = {
        'monitor': 'val_loss',
        'save_best_only': False,
        'save_weights_only': True,
        'mode': 'auto'
    }

    default_early_stopping_kwargs = {
        'monitor': 'acc',
        'mode': 'max',

    }

    default_tensorboard_kwargs = {
        'tensorboard_write_images': True,
        'tensorboard_write_graph': True
    }

    def __init__(
        self,
        dataset_url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        image_shape=(224, 224, 3),
        model_url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
        epochs=50,
        log_dir='job-output/flowers-{}'.format(time()),
        early_stopping_kwargs=default_early_stopping_kwargs,
        tensorboard_kwargs=default_tensorboard_kwargs,
        model_checkpoint_kwargs=default_model_checkpoint_kwargs


    ):
        self.epochs = epochs
        self.log_dir = log_dir
        self.early_stopping_kwargs = {
            **self.default_early_stopping_kwargs, **early_stopping_kwargs}
        self.tensorboard_kwargs = {
            **self.default_tensorboard_kwargs, **tensorboard_kwargs}
        self.model_checkpoint_kwargs = {
            **self.default_model_checkpoint_kwargs, **model_checkpoint_kwargs}

        self.dataset_url = dataset_url
        dataset_filename = dataset_url.split('/')[-1]

        self.data_root = tf.keras.utils.get_file(
            dataset_filename, dataset_url, extract=True)
        self.data_root = os.path.join(os.path.dirname(
            self.data_root), dataset_filename.split('.')[0])

        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255)
        self.image_data = self.image_generator.flow_from_directory(
            str(self.data_root), target_size=image_shape[:2])

        self.base = TFHubModelLayer(
            model_url=model_url,
            trainable=True
        )
        self.base.model.trainable = False

        self.model = tf.keras.Sequential([
            self.base.model,
            tf.keras.layers.Dense(
                self.image_data.num_classes, activation='softmax')
        ])

        logging.info(self.model.summary())

    def train(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['acc'])

        steps_per_epoch = np.ceil(
            self.image_data.samples/self.image_data.batch_size)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.log_dir + '/weights.{epoch:02d}.hdf5', **self.model_checkpoint_kwargs),

            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir, write_images=self.tenosrboard_write_images, write_graph=self.ftensorboard_write_graph,
                histogram_freq=1,
            ),
            tf.keras.callbacks.EarlyStopping(self.early_stopping_kwargs)
        ]
        self.model.fit(self.image_data, epochs=self.epochs,
                       steps_per_epoch=steps_per_epoch,
                       callbacks=callbacks
                       )


if __name__ == '__main__':
    model = TFHubTrainer()
    model.train()
