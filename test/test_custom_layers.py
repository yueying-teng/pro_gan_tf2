import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")

from pro_gan.utils import assert_almost_equal
from pro_gan.custom_layers import (
    PixelNormalization,
    get_scale,
    MinibatchStdDev,
    EqualizedDense,
    EqualizedConv2D,
    )


def test_PixelwiseNorm():
    mock_in = tf.random.normal([1, 1, 1, 13])
    normalizer = PixelNormalization()
    print(f"\nNormalizerBlock: {normalizer}")
    mock_out = normalizer(mock_in)

    # check output
    assert mock_out.shape == mock_in.shape
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(mock_out), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(mock_out), tf.float32)) == 0


def test_MinibatchStdDev() -> None:
    mock_in = tf.random.normal([16, 16, 16, 13])
    min_std_dev = MinibatchStdDev()
    print(f"\nMiniBatchStdDevBlock: {min_std_dev}")
    mock_out = min_std_dev(mock_in)

    # check output
    assert mock_out.shape[-1] == mock_in.shape[-1] + 1
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(mock_out), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(mock_out), tf.float32)) == 0


def test_get_scale():
    shape = (3, 4, 5)
    gain = np.sqrt(2)
    scale = get_scale(shape, gain)

    assert_almost_equal(scale, 0.1825742, error_margin=1e-1)


def test_EqualizedDense():
    mock_in = tf.random.normal([32, 13])
    equalized_dense = EqualizedDense(units=52)
    print(f"Equalized dense block: {equalized_dense}")
    mock_out = equalized_dense(mock_in)

    print([var.name for var in equalized_dense.trainable_variables])

    # check output
    assert mock_out.shape == (32, 52)
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(mock_out), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(mock_out), tf.float32)) == 0
    # check the weight's scale
    print(np.std(equalized_dense.get_weights()[1]))
    assert_almost_equal(np.std(equalized_dense.get_weights()[1]), 1, error_margin=1e-1)


def test_EqualizedConv2D():
    mock_in = tf.random.normal([32, 16, 16, 21])
    equalized_conv2d = EqualizedConv2D(filters=3, kernel_size=(3, 3), padding="SAME")
    print(f"Equalized conv block: {equalized_conv2d}")
    mock_out = equalized_conv2d(mock_in)

    print([var.name for var in equalized_conv2d.trainable_variables])

    # check output
    assert mock_out.shape == (32, 16, 16, 3)
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(mock_out), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(mock_out), tf.float32)) == 0
    # check the weight's scale
    print(np.std(equalized_conv2d.get_weights()[0]))
    assert_almost_equal(np.std(equalized_conv2d.get_weights()[0]), 1, error_margin=1e-1)


def main():
    test_get_scale()
    test_MinibatchStdDev()
    test_PixelwiseNorm()
    test_EqualizedDense()
    test_EqualizedConv2D()


if __name__ == "__main__":
    main()
