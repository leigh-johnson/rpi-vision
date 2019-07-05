# Python
import logging
import os
from time import time
# lib
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import PIL.Image as Image

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
        self.classifier = tf.keras.Sequential([self.model])

        labels_filename = labels_url.split(
            '/')[-1]
        labels_path = tf.keras.utils.get_file(labels_filename, labels_url)
        self.labels = np.array(open(labels_path).read().splitlines())

    def predict(self, top_k=5, image_url='https://i.ibb.co/89YXhK1/Nine-banded-Armadillo.jpg', image_filename='armadillo.jpg'):

        image_data = tf.keras.utils.get_file(image_filename, image_url)
        image_data = Image.open(image_data).resize(self.input_shape[:2])
        image_data = np.array(image_data)/255.0
        image_data = np.expand_dims(image_data, axis=0)
        result = self.classifier.predict(image_data)

        labels_idxs = np.argsort(result[0])[-top_k:]
        labels_idxs = np.flip(labels_idxs)
        predictions = [
            {'label': self.labels[i],
             'prediction': result[0][i],
             'label_idx': i
             }
            for i in labels_idxs]
        return predictions


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
        'write_images': True,
        'write_graph': True,
        'histogram_freq': 1,
    }

    def __init__(
        self,
        dataset_url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        image_shape=(224, 224, 3),
        model_url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
        epochs=50,
        batch_size=24,
        log_dir='job-output/flowers-{}'.format(time()),
        early_stopping_kwargs=default_early_stopping_kwargs,
        tensorboard_kwargs=default_tensorboard_kwargs,
        model_checkpoint_kwargs=default_model_checkpoint_kwargs


    ):
        self.batch_size = batch_size
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
            str(self.data_root), target_size=image_shape[:2],
            batch_size=self.batch_size
        )

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

    def fit(self):
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
                log_dir=self.log_dir, **self.tensorboard_kwargs
            ),
            tf.keras.callbacks.EarlyStopping(self.early_stopping_kwargs)
        ]
        self.model.fit(self.image_data, epochs=self.epochs,
                       steps_per_epoch=steps_per_epoch,
                       callbacks=callbacks
                       )


if __name__ == '__main__':
    # model = TFHubTrainer()
    # model.train()
    model = TFHubModelLayer()
    model.predict()
