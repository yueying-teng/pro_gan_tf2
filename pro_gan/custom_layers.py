import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense


class PixelNormalization(tf.keras.layers.Layer):
    """
    To disallow the scenario where the magnitudes in the generator and discriminator
    spiral out of control as a result of competition, normalization the feature vector
    in each pixel after each convolution layer.

    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    """

    def __init__(self, name=None):
        super(PixelNormalization, self).__init__(name=name)

    def call(self, inputs):
        return inputs * tf.math.rsqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1.0e-8)

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStdDev(tf.keras.layers.Layer):
    """
    GANs have a tendency to capture only a subset of the variation found in training data.
    Computing feature statistics not only from individual images but also across the mini batch
    encourages the mini batch of generated and training images to show similar statistics.
    - Compute the standard deviation for each feature in each spatial location over the mini batch.
    - average these estimates over all features and spatial locations to arrive at a single value.
    - replicate the value and concatenate it to all spatial location and over the mini batch,
      yielding one additional feature map(constant)
    - this layer could be inserted anywhere in the discriminator, but the authors experiments
      found that inserting it at the end performs the best.

    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127

    Args:
        group_size: a integer number, minibatch must be divisible by (or smaller than) group_size.

    Returns: inputs appended with standard deviation constant map
    """

    def __init__(self, group_size=1, name=None):
        super(MinibatchStdDev, self).__init__(name=name)
        self.group_size = group_size

    def call(self, inputs):
        batch_size, height, width, channels = inputs.shape
        if batch_size > self.group_size:
            assert batch_size % self.group_size == 0, (
                f"batch_size {batch_size} should be "
                f"perfectly divisible by group_size {self.group_size}"
            )
            group_size = self.group_size
        else:
            group_size = batch_size

        y = tf.reshape(inputs, [group_size, -1, height, width, channels]) # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                                        # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)                     # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                          # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                             # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)              # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)                                      # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, height, width, 1])                    # [NHW1]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=-1)                            # [NHWC]  Append as new fmap.

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        return (batch_size, height, width, channels + 1)


class Bias(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super(Bias, self).__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias_weights", shape=(self.units,), initializer="zeros", trainable=True
            )

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


def get_scale(shape, gain):
    """
    determine the scale factor for runtime normalization

    - used in Equalized layers
    Args:
        shape: shape of the kernels
        gain: used in He’s initializer: gain / np.sqrt(fan_in)
    Returns:
        per-layer normalization constant
    """
    shape = np.asarray(shape)
    fan_in = np.prod(shape)
    return gain / np.sqrt(fan_in)


class WeightScaling(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(WeightScaling, self).__init__(name=name, **kwargs)

    def call(self, inputs, wscale):
        """
        reweights the input by the scale factor, wscale
        """
        inputs = tf.cast(inputs, tf.float32)
        wscale = tf.cast(wscale, tf.float32)
        return inputs * wscale

    def compute_output_shape(self, input_shape):
        return input_shape


class EqualizedConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        activation=None,
        gain=np.sqrt(2),
        padding="SAME",
        name=None,
        **kwargs,
        ):

        super(EqualizedConv2D, self).__init__(name=name,  **kwargs)
        self.conv2d = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
            padding=padding,
            activation=None,
            dtype=tf.float32,
            )
        self.bias = Bias(units=filters)
        self.kernel_size = kernel_size
        self.weight_scaling = WeightScaling()
        self.gain = gain
        self.activation = activation

    def call(self, inputs):
        """
        here kernel weights rescaling is done on conv2d's output, and before the
        none linear activation, which is the same as applying the scale on conv2d's kernel
        first then sliding across the input
        i.e. activation((x * k) * scale) == activation(x * (k * scale))
             x: input, k: kernel

        TODO: double check if this equality holds
        https://github.com/tensorflow/gan/blob/696f06c49fd598fa3397039a28e597b0b26c43ed/tensorflow_gan/examples/progressive_gan/layers.py#L184
        """
        in_filters = inputs.shape.as_list()[-1]
        scale = get_scale(shape=(self.kernel_size[0], self.kernel_size[1], in_filters), gain=self.gain)
        x = self.conv2d(inputs)
        x = self.weight_scaling(x, scale)
        x = self.bias(x)
        if self.activation:
            x = self.activation(x)
        return x


class EqualizedDense(tf.keras.layers.Layer):
    def __init__(self, units, gain=np.sqrt(2), activation=None, name=None, **kwargs):
        super(EqualizedDense, self).__init__(name=name, **kwargs)
        self.bias = Bias(units=units)
        self.dense = Dense(
            units=units,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
            activation=None,
            dtype=tf.float32,
            )
        self.weight_scaling = WeightScaling()
        self.gain = gain
        self.activation = activation

    def call(self, inputs):
        in_filters = inputs.shape.as_list()[-1]
        scale = get_scale(shape=(in_filters), gain=self.gain)
        x = self.dense(inputs)
        x = self.weight_scaling(x, scale)
        x = self.bias(x)
        if self.activation:
            x = self.activation(x)
        return x


def update_average(model, ema):
    """
    The average() method gives access to the shadow variables.
    It allows you to use the moving averages in place of the last trained values for evaluations,
    by loading the moving averages into your model via var.assign(ema.average(var)).

    assign to your model's variables the shadow variable
    """
    for var in model.trainable_variables:
        var.assign(ema.average(var))
