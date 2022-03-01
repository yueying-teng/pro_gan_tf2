import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")

from pro_gan.modules import (
    # ConDisFinalBlock,
    DisFinalBlock,
    GenInitialBlock,
    )
from pro_gan.networks import nf


def test_GenInitialBlock():
    batch_size, latent_size = 2, 512
    mock_init_generator = GenInitialBlock(latent_size, nf(1), True)
    mock_latent = tf.random.normal([batch_size, latent_size])

    output_features = mock_init_generator(mock_latent)
    mock_init_generator.build_graph().summary()
    assert output_features.shape == (batch_size, 4, 4, nf(1))
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(output_features), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(output_features), tf.float32)) == 0


def test_DisFinalBlock():
    batch_size, latent_size = 2, 512
    mock_dis_final_block = DisFinalBlock(nf(1), latent_size, True)
    mock_input = tf.random.normal([batch_size, 4, 4, 512])

    output = mock_dis_final_block(mock_input)
    assert output.shape == (batch_size, 1)
    assert tf.reduce_sum(tf.cast(tf.math.is_nan(output), tf.float32)) == 0
    assert tf.reduce_sum(tf.cast(tf.math.is_inf(output), tf.float32)) == 0


def main():
    test_GenInitialBlock()
    test_DisFinalBlock()


if __name__ == "__main__":
    main()
