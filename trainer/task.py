from keras.applications.mobilenetv2 import MobileNetV2
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from datetime import datetime
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage

storage_client = storage.Client()

bucket = storage_client.get_bucket('raspberry-pi-vision')

# initialize convolutional base
conv_base = MobileNetV2(
    weights='imagenet',
    input_shape=(224,224, 3),
    include_top=False # exclude the densely connected classifer, which sits on top of hte convolutional network
)
conv_base.trainable = False

# initialize Sequential model and Dense players
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#initialize image data generators

# augment training data to produce a model capable of handling data variations
train_datagen = ImageDataGenerator(
    rescale=1./255, # rescale to target values between 0 and 255 (default between 0 and 1)
    rotation_range=40, # train on variations rotated up to 40 degrees
    width_shift_range=0.9, # train using variations off-center on x-axis by factor of 0.9
    height_shift_range=0.9, # train using variations off-center on y-axis by a factor of 0.9
    shear_range=0.2, # train using variations sheared/warped by a factor of 0.2
    zoom_range=0.9, # train using variations zoomed by a factor of 0.9
    horizontal_flip=True, # x-axis flip
    vertical_flip=True # y-axis flip
)

test_datagen = ImageDataGenerator(
    rescale=1./255, # rescale to target values between 0 and 255 (default between 0 and 1)
)

# walk through './data/shapes' and load filenames into a dataframe with labels

# read from fs
# root, dirs, files = next(os.walk('./data/shapes/'))

# read from gcp
files = bucket.list_blobs(prefix='shapes/training-data/')


samples_df = pd.DataFrame([
    {
        'label': file.split('_')[0], # filename format '<label>_<int>.png'
        'filename': file
    } for file in files
])

train_df, validation_df = train_test_split(samples_df, test_size=0.25)

train_df.reset_index(inplace=True, drop=True)
validation_df.reset_index(inplace=True, drop=True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='./data/shapes/',
    batch_size=len(train_df),
    class_mode='binary', # use binary labels for binary_crossentropy loss calculations
    target_size=(224,224),
    y_col='label',
)


validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory='./data/shapes/',
    batch_size=len(validation_df),
    class_mode='binary', # use binary labels for binary_crossentropy loss calculations
    target_size=(224,224),
    y_col='label',
    classes=['square', 'circle', 'triangle']

)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5), # what is this?
    metrics=['acc'] # accuracy
)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[time_callback]
)

model.save('./shapes_model.h5')
blob = bucket.blob('shapes/models/{0}'.format((datetime.utcnow()))
blob.upload_from_filename(filename='./shapes_model.h5')
