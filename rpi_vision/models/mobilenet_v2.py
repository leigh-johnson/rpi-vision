# Python
import logging
# lib
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

logging.basicConfig()


class MobileNetV2Base():
    def __init__(self,
                 input_shape=None,
                 alpha=1.0,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 ):

        self.input_shape = input_shape

        self.model_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
            alpha=alpha,
            classes=classes,
            include_top=include_top,
            input_shape=input_shape,
            input_tensor=input_tensor,
            pooling=pooling,
            weights=weights,
        )
        logging.info(self.model_base.summary())

    def predict(self, frame):
        # expand 3D RGB frame into 4D batch
        sample = np.expand_dims(frame, axis=0)
        processed_sample = preprocess_input(sample.astype(np.float32))
        features = self.model_base.predict(processed_sample)
        decoded_features = decode_predictions(features)
        return decoded_features

    def tflite_convert(self, output_dir='includes/', filename='mobilenet_v2.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        tflite_model = converter.convert()
        if output_dir and filename:
            with open(output_dir + filename, 'wb') as f:
                f.write(tflite_model)
        return tflite_model


if __name__ == '__main__':
    mobilenetv2 = MobileNetV2()
    mobilenetv2.tflite_convert()
