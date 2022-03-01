import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt

from pro_gan.networks import Discriminator, Generator


def adjust_dynamic_range(
    data,
    drange_in = (-1, 1),
    drange_out = (-1, 1),
    ):
    if drange_in != drange_out:
        scale = (tf.cast(drange_out[1], tf.float32) - tf.cast(drange_out[0], tf.float32)) / (
            tf.cast(drange_in[1], tf.float32) - tf.cast(drange_in[0], tf.float32))
        bias = tf.cast(drange_out[0], tf.float32) - tf.cast(drange_in[0], tf.float32) * scale
        data = data * scale + bias

    return tf.clip_by_value(tf.convert_to_tensor(data), drange_out[0], drange_out[1])


def get_logging_dirs(save_dir, current_time):
    """
    return the dirs for saving model checkpoints and feedback images
    during training as well as a tensorboard writer

    Args:
        save_dir: target dir to save those mentioned above
    """
    # for saving generated sample images during training
    train_log_dir = save_dir / "logs" / f"{current_time}"

    tf_board_dir = save_dir / "logs" / f"{current_time}" / "tensorboard"
    tf_board_dir.mkdir(parents=True, exist_ok=True)
    tf_board_writer = tf.summary.create_file_writer(str(tf_board_dir))

    model_dir = save_dir / "models" / f"{current_time}"
    model_dir.mkdir(parents=True, exist_ok=True)

    return tf_board_writer, train_log_dir, model_dir


def get_layer_weights(model):
    """
    collect the model's layer weight in a dictionary
    with layer name as dictionary keys
    """
    dict = {}
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            dict[layer.name] = np.array(w, dtype=object)

    return dict


def transfer(model, layer_weights):
    """
    load the weights stored in layer_weights, a dictionary,
    to the given model's layers

    Args:
        model: a model whose weights will be reset
        layer_weights: a dictionary with layer name as keys and
                       weights as values
    Returns:
        model with layer weights reset
    """
    for layer in model.layers:
        print(layer.name)
        if layer.name in layer_weights:
            print(f"transfering weights to {layer.name}")
            layer.set_weights(layer_weights[layer.name])

    return model


def create_grid(samples, img_file, figure_size=(20, 20)):
    """
    utility function to create a grid of GAN samples

    Args:
        samples: generated samples for feedback
        scale_factor: factor for upscaling the image
        img_file: name of file to write
    Returns: None (saves a file)
    """
    samples = adjust_dynamic_range(
        samples, drange_in = (-1, 1), drange_out = (0, 1)
        )
    fig = plt.figure(figsize=figure_size)
    row_num = int(np.sqrt(len(samples)))
    for i in range(samples.shape[0]):
        _ = plt.subplot(row_num, row_num, i + 1)
        plt.imshow(samples[i])
    plt.savefig(img_file)
    plt.close(fig)


def progressive_downsample_batch(total_depth, real_batch, depth, alpha):
    """
    downsample the original images for the given depth
    Args:
        total_depth: defines the final output resolution once all
                     models are progressively trained
        real_batch: batch of real samples
        depth: depth at which training is happening
        alpha: current value of the fader alpha

    Returns: modified real batch of samples
    """

    down_sample_factor = int(2 ** (total_depth - depth))
    prior_downsample_factor = int(2 ** (total_depth - depth + 1))
    ds_real_samples = AveragePooling2D(
        pool_size=down_sample_factor, strides=down_sample_factor
        )(real_batch)

    if depth > 2:
        prior_ds_real_samples = AveragePooling2D(
            pool_size=prior_downsample_factor,
            strides=prior_downsample_factor,
            )(real_batch)
        prior_ds_real_samples = UpSampling2D(size=2)(prior_ds_real_samples)
    else:
        prior_ds_real_samples = ds_real_samples

    # real samples are a linear combination of ds_real_samples and prior_ds_real_samples
    real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

    return tf.convert_to_tensor(real_samples, dtype=tf.float32)


def load_discriminator(model_dir, depth, last_epoch, latent_size, total_depth, real_images, use_eql=True):
    """
    load the specified discriminator with the given model checkpoints dir

    Args:
        model_dir: pathlib.Path("./models/20220215")
                   model weight files have the following naming pattern
                   f"gen_depth_{depth}_epoch_{last_epoch}"
        depth: required depth of the network
               used to determine the architecture
               i.e. input is of resolution 2 ** depth
        real_images: a batch of real data with resolution 2 ** depth
                     used for training models at the current depth
        total_depth: defines the final generated otuput resolution, 2 ** total_depth,
                    once all models are progressively trained
        use_eql: whether to use Equalized Conv2d and Dense
    Returns:
        discriminator model loaded with weights from checkpoint path
        model_dir / f"gen_depth_{depth}_epoch_{last_epoch}"
    """

    dis_dir = model_dir / f"dis_depth_{depth}_epoch_{last_epoch}"

    dis = Discriminator(depth=depth, latent_size=latent_size, use_eql=use_eql)
    # feed the model input before loading the weights/checkpoints to properly initialize shapes
    downsampled_images = progressive_downsample_batch(total_depth, real_images, depth, 1.0)
    dis(downsampled_images, depth, 1.0)
    dis.load_weights(dis_dir)

    return dis


def load_generator(model_dir, depth, last_epoch, latent_size, use_eql=True):
    """
    load the specified generator with the given model checkpoints dir

    Args:
        model_dir: pathlib.Path("./models/20220215")
                   model weight files have the following naming pattern
                   f"gen_depth_{depth}_epoch_{last_epoch}"
        depth: required depth of the network
               i.e. output is of resolution 2 ** depth
        use_eql: whether to use Equalized Conv2d and Dense
    Returns:
        generator model loaded with weights from checkpoint path
        model_dir / f"gen_depth_{depth}_epoch_{last_epoch}"
    """

    gen_dir = model_dir / f"gen_depth_{depth}_epoch_{last_epoch}"

    gen = Generator(depth=depth, latent_size=latent_size, use_eql=use_eql)
    # feed the model input before loading the weights/checkpoints to properly initialize shapes
    gan_input = tf.convert_to_tensor(tf.random.normal([1, latent_size]), dtype=tf.float32)
    gen(gan_input, depth, 1.0)
    gen.load_weights(gen_dir)

    return gen


def assert_almost_equal(x, y, error_margin):
    assert np.abs(x - y) <= error_margin
