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
        self.tflite_interpreter = None

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

    def tflite_convert(self, output_dir='includes/', output_filename='mobilenet_v2.tflite', from_keras_model_file='includes/mobilenet_v2.h5'):
        # @todo TFLiteConverter.from_keras_model() is only available in the tf-nightly-2.0-preview build right now
        # https://groups.google.com/a/tensorflow.org/forum/#!searchin/developers/from_keras_model%7Csort:date/developers/Mx_EaHM1X2c/rx8Tm-24DQAJ
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        converter = tf.lite.TFLiteConverter.from_keras_model_file(self.model_base)
        tflite_model = converter.convert()
        if output_dir and filename:
            with open(output_dir + output_filename, 'wb') as f:
                f.write(tflite_model)
                logger.info('Wrote {}'.format(output_dir + filename))
        return tflite_model

    def init_tflite_interpreter(self, model_path='includes/mobilenet_v2.tflite'):
        '''
            https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/lite/Interpreter
            This makes the TensorFlow Lite interpreter accessible in Python. 
            It is possible to use this interpreter in a multithreaded Python environment, but you must be sure to call functions of a particular instance from only one thread at a time. 
            So if you want to have 4 threads running different inferences simultaneously, create an interpreter for each one as thread-local data. 
            Similarly, if you are calling invoke() in one thread on a single interpreter but you want to use tensor() on another thread once it is done
            you must use a synchronization primitive between the threads to ensure invoke has returned before calling tensor().

        '''
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.tflite_interpreter.allocate_tensors()
        logging.info('Initialized tflite Python interpreter \n',
                     self.tflite_interpreter)

        self.tflite_input_details = self.tflite_interpreter.get_input_details()
        logging.info('tflite input details \n', self.tflite_input_details)

        self.tflite_output_details = self.tflite_interpreter.get_output_details()
        logging.info('tflite output details \n',
                     self.tflite_output_details)

        return self.tflite_interpreter

    def tflite_predict(self, input_data, input_shape=None):
        if not self.tflite_interpreter:
            self.tflite_convert()
            self.init_tflite_interpreter()

        self.tflite_interpreter.set_tensor(
            self.tflite_input_details[0]['index'], input_data)
        self.tflite_interpreter.invoke()

        output_data = self.tflite_interpreter.get_tensor(
            self.tflite_output_details[0]['index'])
        return output_data


if __name__ == '__main__':
    mobilenetv2 = MobileNetV2()
    mobilenetv2.tflite_convert()
