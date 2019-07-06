# Python
import logging
import os
from time import time
# lib
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import PIL.Image as Image


class TFHubModel(object):
    '''
        Google colab examples: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb
    '''

    def __init__(
        self,
        input_shape=(224, 224, 3),
        model_url='https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1',
        trainable=False
    ):

        self.model_url = model_url
        self.input_shape = input_shape

        self.model = hub.KerasLayer(
            model_url, input_shape=input_shape, trainable=trainable)
        self.classifier = tf.keras.Sequential([self.model])

    def predict(self, frame, top_k=1):

        # image_data = tf.keras.utils.get_file(image_filename, image_url)
        # image_data = Image.open(image_data).resize(self.input_shape[:2])
        image_data = np.array(frame)/255.0
        result = self.classifier.predict(image_data)

        # labels_idxs = np.argsort(result[0])[-top_k:]
        # labels_idxs = np.flip(labels_idxs)
        # predictions = [
        #     {'label': self.labels[i],
        #      'prediction': result[0][i],
        #      'label_idx': i
        #      }
        #     for i in labels_idxs]
        return predictions


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
