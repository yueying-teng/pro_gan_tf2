import tensorflow as tf
import sys
sys.path.append("../")

from pro_gan.networks import Discriminator, Generator


def test_Generator():
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024
    mock_latent = tf.random.normal([batch_size, latent_size])

    for res_log2 in range(2, depth + 1):
        mock_generator = Generator(
            depth=res_log2, num_channels=num_channels, latent_size=latent_size
            )
        print(f"Generator Network:\n{mock_generator}")
        rgb_images = mock_generator(mock_latent, depth=res_log2, alpha=1)
        print(f"generated RGB output shape at depth {res_log2}: {rgb_images.shape}")

        assert rgb_images.shape == (
            batch_size,
            2 ** res_log2,
            2 ** res_log2,
            num_channels,
        )
        assert tf.reduce_sum(tf.cast(tf.math.is_nan(rgb_images), tf.float32)) == 0
        assert tf.reduce_sum(tf.cast(tf.math.is_inf(rgb_images), tf.float32)) == 0


def test_DiscriminatorUnconditional():
    batch_size, latent_size = 2, 512
    num_channels = 3
    depth = 10  # resolution 1024 x 1024

    mock_inputs = [
        tf.random.normal([batch_size, 2 ** stage, 2 ** stage, num_channels])
        for stage in range(2, depth + 1)
        ]

    for res_log2 in range(2, depth + 1):
        mock_discriminator = Discriminator(
            depth=res_log2, num_channels=num_channels, latent_size=latent_size
            )
        print(f"Discriminator Network:\n{mock_discriminator}")
        mock_input = mock_inputs[res_log2 - 2]
        print(f"RGB input image shape at depth {res_log2}: {mock_input.shape}")
        score = mock_discriminator(mock_input, depth=res_log2, alpha=1)

        assert score.shape == (batch_size, 1)
        assert tf.reduce_sum(tf.cast(tf.math.is_nan(score), tf.float32)) == 0
        assert tf.reduce_sum(tf.cast(tf.math.is_inf(score), tf.float32)) == 0


# def test_DiscriminatorConditional():
#     batch_size, latent_size = 2, 512
#     num_channels = 3
#     depth = 10  # resolution 1024 x 1024
#     mock_discriminator = Discriminator(depth=depth, num_channels=num_channels, num_classes=10)
#     mock_inputs = [
#         tf.random.normal([batch_size, 2 ** stage, 2 ** stage, num_channels])
#         for stage in range(2, depth + 1)
#     ]
#     mock_labels = np.array([3, 7])

#     print(f"Discriminator Network:\n{mock_discriminator}")
#     for res_log2 in range(2, depth + 1):
#         mock_input = mock_inputs[res_log2 - 2]
#         print(f"RGB input image shape at depth {res_log2}: {mock_input.shape}")
#         score = mock_discriminator(
#             mock_input, depth=res_log2, alpha=1, labels=mock_labels
#         )
#         assert score.shape == (batch_size,)
#         assert tf.reduce_sum(tf.cast(tf.math.is_nan(score), tf.float32)) == 0
#         assert tf.reduce_sum(tf.cast(tf.math.is_inf(score), tf.float32)) == 0


def main():
    test_DiscriminatorUnconditional()
    test_Generator()


if __name__ == "__main__":
    main()
