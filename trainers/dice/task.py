# python
import os
from datetime import datetime
import argparse
import tarfile
import urllib
from time import time

# lib
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from google.cloud import storage
import trainers
import pandas as pd

# app
from trainers.common.keras_preprocessing_patched import ImageDataGenerator
from trainers.common.callback import TimeHistory, GCSModelCheckpoint, GCSTensorBoard


# inspired by https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

# replace with path to your bucket!
BATCH_SIZE = 16

IMG_HEIGHT, IMG_WIDTH = 480, 480


NUM_TRAIN_SAMPLES = 14290
NUM_VALIDATION_SAMPLES = 210

if K.image_data_format() == "channels_first":
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)


def compose_dataframe(path, path_suffix):
    # walk through './data/shapes' and load filenames into a dataframe with labels
    # read from fs
    label_dirs = next(os.walk(path + path_suffix))[1]

    samples = []
    for label_path in label_dirs:
        files = os.listdir(path + path_suffix + "/" + label_path)
        labels = tuple(label_path.split("_"))
        samples.append([(file, labels, label_path) for file in files])

    samples = [item for sublist in samples for item in sublist]

    df = pd.DataFrame(
        [
            {"label": labels, "filename": label_path + "/" + filename}
            for (filename, labels, label_path) in samples
            if filename.endswith(".jpg") and not filename.startswith("._")
        ]
    )

    return df


def main():

    GCS_BUCKET = "gs://raspberry-pi-vision/dice/"
    REMOTE_DATA_PATH = "dice/"

    TRAINING_TARBALL = (
        "https://storage.googleapis.com/raspberry-pi-vision/dice/data.tar.gz"
    )

    MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
    LOCAL_DATA_PATH = MODULE_PATH + "/data/"

    # only download and extract tar'd data if raspberry-pi-vision/trainers/dice/data/ does not exist
    if not os.path.isdir(LOCAL_DATA_PATH):
        file_path = MODULE_PATH + "/data.tar.gz"
        if not os.path.isfile(file_path):
            print("Downloading {0}".format(TRAINING_TARBALL))
            file = urllib.request.URLopener()
            file.retrieve(TRAINING_TARBALL, file_path)
        print("Extracting {0}".format(file_path))
        tar = tarfile.open(file_path)
        tar.extractall(MODULE_PATH)
        tar.close()

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(25))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_df = compose_dataframe(LOCAL_DATA_PATH, "train/")
    validation_df = compose_dataframe(LOCAL_DATA_PATH, "valid/")

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        LOCAL_DATA_PATH + "train/",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        y_col="label",
        classes=list(set(train_df["label"])),
    )

    validation_generator = test_datagen.flow_from_dataframe(
        validation_df,
        LOCAL_DATA_PATH + "valid/",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        y_col="label",
        classes=list(set(validation_df["label"])),
    )

    # report step + epoch progress
    time_callback = TimeHistory()

    # checkpoint
    checkpoint_filepath = (
        "{0}_weights_".format(trainers.__version__) + "{epoch:02d}-{val_acc:.2f}.hdf5"
    )

    checkpoint_remotepath = (
        REMOTE_DATA_PATH
        + "checkpoint/{}/".format(trainers.__version__)
        + checkpoint_filepath
    )
    checkpoint_callback = GCSModelCheckpoint(
        checkpoint_filepath,
        checkpoint_remotepath,
        bucket="raspberry-pi-vision",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    tensorboard_callback = GCSTensorBoard(
        log_dir="logs/{}_{}".format(trainers.__version__, datetime.utcnow()),
        remote_log_dir=GCS_BUCKET + "logs/{}".format(trainers.__version__),
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_VALIDATION_SAMPLES // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=NUM_VALIDATION_SAMPLES // BATCH_SIZE,
        callbacks=[time_callback, checkpoint_callback, tensorboard_callback],
    )

    model.save_weights(LOCAL_DATA_PATH + "weights_" + trainers.__version__ + ".h5")
    model.save(LOCAL_DATA_PATH + "model_" + trainers.__version__ + ".h5")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("raspberry-pi-vision")
    w_blob = bucket.blob(
        REMOTE_DATA_PATH
        + "models/final_weights_{0}_{1}".format(trainers.__version__, datetime.utcnow())
    )
    w_blob.upload_from_filename(
        filename=LOCAL_DATA_PATH + "weights_" + trainers.__version__ + ".h5"
    )

    m_blob = bucket.blob(
        REMOTE_DATA_PATH
        + "models/model_{0}_{1}".format(trainers.__version__, datetime.utcnow())
    )
    m_blob.upload_from_filename(
        filename=LOCAL_DATA_PATH + "model_" + trainers.__version__ + ".h5"
    )


if __name__ == "__main__":
    main()

