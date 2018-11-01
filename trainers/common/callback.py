import time
import os
import warnings
import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from google.cloud import storage
from subprocess import call


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class GCSModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        remote_path: string, Google Cloud Storage blob path to save file remotely
        bucket: string, name of Google Cloud Storage bucket
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        remote_path,
        bucket=None,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        super(GCSModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.get_bucket(bucket)
        self.remote_path = remote_path

        if not bucket:
            raise ValueError(
                "`bucket` is required and should be the name of a Google Cloud Storage bucket"
            )
        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                "ModelCheckpoint mode %s is unknown, "
                "fallback to auto mode." % (mode),
                RuntimeWarning,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            remote_path = self.remote_path.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                            blob = self.bucket.blob(remote_path)
                            blob.upload_from_filename(filename=filepath)
                        else:
                            self.model.save(filepath, overwrite=True)
                            blob = self.bucket.blob(remote_path)
                            blob.upload_from_filename(filename=filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                    blob = self.bucket.blob(remote_path)
                    blob.upload_from_filename(filename=filepath)
                else:
                    self.model.save(filepath, overwrite=True)
                    blob = self.bucket.blob(remote_path)
                    blob.upload_from_filename(filename=filepath)


class GCSTensorBoard(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/programmers_guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """

    def __init__(
        self,
        log_dir="./logs",
        remote_log_dir=None,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq="epoch",
    ):
        super(GCSTensorBoard, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError(
                "You need the TensorFlow module installed to " "use TensorBoard."
            )

        if K.backend() != "tensorflow":
            if histogram_freq != 0:
                warnings.warn(
                    "You are not using the TensorFlow backend. "
                    "histogram_freq was set to 0"
                )
                histogram_freq = 0
            if write_graph:
                warnings.warn(
                    "You are not using the TensorFlow backend. "
                    "write_graph was set to False"
                )
                write_graph = False
            if write_images:
                warnings.warn(
                    "You are not using the TensorFlow backend. "
                    "write_images was set to False"
                )
                write_images = False
            if embeddings_freq != 0:
                warnings.warn(
                    "You are not using the TensorFlow backend. "
                    "embeddings_freq was set to 0"
                )
                embeddings_freq = 0

        self.log_dir = log_dir
        self.remote_log_dir = remote_log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        if update_freq == "batch":
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def set_model(self, model):
        self.model = model
        if K.backend() == "tensorflow":
            self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(":", "_")
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads and weight in layer.trainable_weights:
                        grads = model.optimizer.get_gradients(model.total_loss, weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == "IndexedSlices"

                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads
                        ]
                        tf.summary.histogram(
                            "{}_grad".format(mapped_weight_name), grads
                        )
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1, shape[0], shape[1], 1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == "channels_last":
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0], shape[1], shape[2], 1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1, shape[0], 1, 1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, "output"):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram(
                                "{}_out_{}".format(layer.name, i), output
                            )
                    else:
                        tf.summary.histogram("{}_out".format(layer.name), layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq and self.embeddings_data is not None:
            self.embeddings_data = standardize_input_data(
                self.embeddings_data, model.input_names
            )

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [
                    layer.name
                    for layer in self.model.layers
                    if type(layer).__name__ == "Embedding"
                ]
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = np.prod(embedding_input.shape[1:])
                    embedding_input = tf.reshape(
                        embedding_input, (step, int(embedding_size))
                    )
                    shape = (self.embeddings_data[0].shape[0], int(embedding_size))
                    embedding = tf.Variable(
                        tf.zeros(shape), name=layer.name + "_embedding"
                    )
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(
                        embedding[batch_id : batch_id + step], embedding_input
                    )
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {
                    layer_name: self.embeddings_metadata
                    for layer_name in embeddings_vars.keys()
                }

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError(
                "If printing histograms, validation_data must be "
                "provided, and cannot be a generator."
            )
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError(
                "To visualize embeddings, embeddings_data must " "be provided."
            )
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (
                    self.model.inputs + self.model.targets + self.model.sample_weights
                )

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i : i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i : i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {
                            _input: embeddings_data[idx][batch]
                            for idx, _input in enumerate(self.model.input)
                        }
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(
                        self.sess,
                        os.path.join(self.log_dir, "keras_embedding.ckpt"),
                        epoch,
                    )
                    blob = self.bucket.blob(self.remote_path)
                    blob.upload_from_filename(
                        filename=os.path.join(self.log_dir, "keras_embedding.ckpt")
                    )

                    i += self.batch_size

        if self.update_freq == "epoch":
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ["batch", "size"]:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()
        call(["gsutil", "-m", "rsync", "-r", self.log_dir, self.remote_log_dir])

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != "epoch":
            self.samples_seen += logs["size"]
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen
