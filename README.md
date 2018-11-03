# RPI Vision

Deep object detection on a Raspberry Pi using Tensorflow & Keras.

### Materials

* Raspberry Pi 3 Model B
* SD card 8+ GB
* 3.5" 480 x 320 TFT/SPI screen (XPT2046  controller)

### Install Dependencies (on Raspberry Pi)

* [Install Raspbian](https://www.raspberrypi.org/documentation/installation/installing-images/README.md)
* [Configure WiFi (optional, but recommended)](https://www.raspberrypi.org/forums/viewtopic.php?t=191252)
* @todo link to other basic Pi configuration tasks outside the scope of this guide (add authorized SSH keys, disable password, change hostname)
* Install system dependencies

```
sudo apt-get update && \
sudo apt-get upgrade && \
sudo apt-get install git python3-dev python3-pip \
crossbuild-essential-armhf libatlas-base-dev   \
libhdf5-dev libhdf5-serial-dev \ 
libopenjp2-7-dev ibtiff5 build-essential cmake pkg-config && \
sudo pip3 install -U virtualenv
```

```
git clone git@github.com:leigh-johnson/rpi-vision.git
cd rpi-vision
pip install -r rpi.requirements.txt
```

### Install TFT Drivers

**WARNING** these instructions **only** apply to the 3.5" TFT (XPT2046) screen. If you're using a difference size or controller, please refer to the instructions in [LCD-show#README](https://github.com/goodtft/LCD-show).


```
git clone git@github.com:goodtft/LCD-show.git
chmod -R 755 LCD-show
cd LCD-show
sudo ./LCD35-show
```

### Install FBCP

This step is only neccessary if you're using an SPI Display. If you're using an HDMI display, skip this step.

### Updating /boot/config.txt

For better TFT screen performance, add the following to `/boot/config.txt`. Refer to Raspbian's [video options in config.txt](https://www.raspberrypi.org/documentation/configuration/config-txt/video.md) if you're using a different display.

@ todo

### Setup Google Cloud (optional)

@todo


### Running a trainer (GPU Accelerated)

```
pip install -r trainer.requirements.txt
```

@todo API docs


### Training a custom CNN

@todo API docs

### Analyzing via Tensorboard

```
tensorboard --logdir gs://my-gcs-bucket/my-model/logs/
```

### References

* [Training a neural network from little data](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
* [How to easily Detect Objects with Neural Nets](https://medium.com/nanonets/how-to-easily-detect-objects-with-deep-learning-on-raspberrypi-225f29635c74)
* [d4, d6, d8. d10, d20 images](https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images)