from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions

import numpy as np



class MobileNetV2Detector(object):

    def __init__(
      self,
      input_size=None
    ):

      self.conv_base = MobileNetV2(
        weights='imagenet',
        include_top=True, # include the densely connected classifer, which sits on top of hte convolutional network
        #input_shape=(480, 640, 3)
      )
      print(self.conv_base.summary())


    def predict(self, frame):
      # expand 3D RGB frame into 4D "batch"
      sample = np.expand_dims(frame, axis=0)
      processed_sample = preprocess_input(sample.astype(np.float32))
      features = self.conv_base.predict(processed_sample)
      decoded_features = decode_predictions(features)
      return decoded_features


class DiceDetector(object):
    def __init__(
      self,
      input_size,
    ):

      self.input_size = input_size


  
  
