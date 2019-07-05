#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unittest import TestCase
from unittest import mock
import pytest
from rpi_vision.trainer.tfhub import TFHubModelLayer, TFHubTrainer


class TestTFHubTrainer(TestCase):

    @mock.patch('tensorflow.keras.preprocessing.image.ImageDataGenerator')
    @mock.patch('tensorflow.keras.Sequential')
    @mock.patch('rpi_vision.trainer.tfhub.TFHubModelLayer')
    def test_defaults(self, *mocks):
        trainer = TFHubTrainer()
        self.assertEqual(trainer.default_early_stopping_kwargs,
                         trainer.early_stopping_kwargs)
        self.assertEqual(trainer.default_model_checkpoint_kwargs,
                         trainer.model_checkpoint_kwargs)
        self.assertEqual(trainer.default_tensorboard_kwargs,
                         trainer.tensorboard_kwargs)
