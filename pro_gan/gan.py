import os
import timeit
import gc
import copy
import datetime
import timeit
import time
from pathlib import Path
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from pro_gan.custom_layers import update_average
from pro_gan.data_tools import load_dataset
from pro_gan.losses import LSGAN, WganGP
from pro_gan.utils import (
    get_logging_dirs,
    get_layer_weights,
    transfer,
    create_grid,
    load_generator,
    load_discriminator,
    progressive_downsample_batch,
    )
from pro_gan.networks import Discriminator, Generator

seed = 847
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)


class ProGANTrainer:
    def __init__(
        self,
        depth=10, # when all models are trained, generator will output images of resolution 2 ** depth
        latent_size=512, # size of the latent manifold
        gen_learning_rate=0.001,
        dis_learning_rate=0.001,
        use_ema=False,
        ema_beta=0.999,
        save_dir=Path("./train"),
        ):

        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.depth = depth
        self.latent_size = latent_size
        # create the generator and discriminator optimizers
        self.gen_optim = Adam(
            learning_rate=gen_learning_rate,
            beta_1=0,
            beta_2=0.99,
            epsilon=1e-8,
            )
        self.dis_optim = Adam(
            learning_rate=dis_learning_rate,
            beta_1=0,
            beta_2=0.99,
            epsilon=1e-8,
            )
        if self.use_ema:
            # TODO: impelment EMA update of generator weights
            # create the EMA object before the training loop
            self.ema = tf.train.ExponentialMovingAverage(decay=ema_beta)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tf_board_writer, self.train_log_dir, self.model_dir = get_logging_dirs(save_dir, current_time)

    def optimize_discriminator(self):
        """
        performs a single weight update step on discriminator
        force retracing of the graph for each new model
        https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844

        Args:
            gen: generator model
            dis: discriminator model
            loss: the loss function to be used for the optimization
            noise: input noise for sample generation
            real_samples: a batch of real data
            depth: current depth of optimization
            alpha: current alpha for fade-in
            labels: labels for conditional discrimination

        Returns: discriminator loss value
        """

        @tf.function(experimental_relax_shapes=True)
        def apply_grad(
            gen,
            dis,
            loss,
            noise,
            real_samples,
            depth,
            alpha,
            labels=None,
            ):
            with tf.GradientTape() as tape:
                # generate a batch of samples
                fake_samples = gen(noise, depth, alpha, training=True)
                dis_loss = loss.dis_loss(dis, real_samples, fake_samples, depth, alpha, labels=labels)

            # Get the gradients w.r.t the discriminator loss
            dis_gradient = tape.gradient(dis_loss, dis.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.dis_optim.apply_gradients(zip(dis_gradient, dis.trainable_variables))
            return dis_gradient, dis_loss
        return apply_grad

    def optimize_generator(self):
        """
        performs a single weight update step on generator
        force retracing of the graph for each new model
        https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844

        Args:
            loss: the loss function to be used for the optimization
            noise: input noise for sample generation
            depth: current depth of optimization
            alpha: current alpha for fade-in
            labels: labels for conditional discrimination

        Returns: generator loss value
        """

        @tf.function(experimental_relax_shapes=True)
        def apply_grad(
            gen,
            dis,
            loss,
            noise,
            depth,
            alpha,
            labels=None,
            ):
            with tf.GradientTape() as tape:
                # generate fake samples:
                fake_samples = gen(noise, depth, alpha, training=True)
                gen_loss = loss.gen_loss(dis, fake_samples, depth, alpha, labels=labels)

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(gen_loss, gen.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.gen_optim.apply_gradients(zip(gen_gradient, gen.trainable_variables))
            return gen_gradient, gen_loss
        return apply_grad

    def train(
        self,
        data_dir,
        crop_size,
        batch_sizes,
        epochs,
        fade_in_percentages,
        loss_fn=WganGP(),
        num_samples=16,
        start_depth=2,
        feedback_factor=100,
        checkpoint_factor=10,
        ):
        """
        # TODO implement support for conditional GAN here

        Args:
            data_dir: dir of the dataset
            crop_size: size of the cropped images
            batch_sizes: list of batch_sizes for every resolution
            epochs: list of number of epochs to train the network for every resolution
            fade_in_percentages: list of percentages of epochs per resolution to use fade-in
                                 used for fading in the new layer not used for
                                 first resolution, but dummy value is still needed
            loss_fn: loss function used for training
            num_samples: number of samples generated in sample sheet
            start_depth: start training from this depth
            feedback_factor: number of logs per epoch
            checkpoint_factor: save the models after these many epochs (applicable for each resolution).

        Returns: None (Writes multiple files to disk)
        """

        assert (self.depth - 1) == len(batch_sizes), "batch_sizes are not compatible with depth"
        assert (self.depth - 1) == len(epochs), "epochs are not compatible with depth"

        # image saving mechanism
        dummy_data_loader = load_dataset(
            data_dir, num_samples, crop_size, resize_to=int(2 ** self.depth)
            )
        real_images_to_downsample = next(iter(dummy_data_loader))
        fixed_input = tf.random.normal([num_samples, self.latent_size], seed=0)

        # create a global time counter
        global_time = time.time()
        step = 1

        for current_depth in range(start_depth, self.depth + 1):
            current_res = int(2 ** current_depth)
            print(f"\n\nCurrently working on Depth: {current_depth}")
            print("Current resolution: %d x %d" % (current_res, current_res))
            depth_list_index = current_depth - 2
            current_batch_size = batch_sizes[depth_list_index]
            data = load_dataset(
                data_dir, current_batch_size, crop_size, resize_to=int(2 ** self.depth)
                )
            real_downsampled_images_for_render = progressive_downsample_batch(
                self.depth, real_images_to_downsample, current_depth, 1)
            create_grid(
                real_downsampled_images_for_render,
                img_file=self.train_log_dir / f"real_images_depth_{current_depth}.png",
                )

            gen = Generator(depth=current_depth, latent_size=self.latent_size, use_eql=True)
            dis = Discriminator(depth=current_depth, latent_size=self.latent_size, use_eql=True)
            feedback_generator = gen

            if current_depth > start_depth:
                print(f"transferring weights from model with depth {current_depth - 1}")
                print(f" to model with depth {current_depth}")
                gen, dis = self.weight_transfer(
                    gen,
                    dis,
                    current_depth,
                    epochs,
                    self.latent_size,
                    batch_sizes,
                    real_images_to_downsample,
                    )

            gen_apply_grad = self.optimize_generator()
            dis_apply_grad = self.optimize_discriminator()

            ticker = 1
            for epoch in range(1, epochs[depth_list_index] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print(f"\nEpoch: {epoch}")
                total_batches = len(data)

                # compute the fader point
                fader_point = int(
                    (fade_in_percentages[depth_list_index] / 100)
                    * epochs[depth_list_index]
                    * total_batches
                    )

                for i, images in enumerate(data, start=1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1
                    gan_input = tf.convert_to_tensor(
                        tf.random.normal([current_batch_size, self.latent_size]), dtype=tf.float32
                        )
                    downsampled_images = progressive_downsample_batch(self.depth, images, current_depth, alpha)

                    dis_grad, dis_loss = dis_apply_grad(
                        gen, dis,
                        loss_fn, gan_input, downsampled_images,
                        tf.constant(current_depth, tf.int32),
                        tf.constant(alpha, tf.float32),
                        )
                    with self.tf_board_writer.as_default():
                        for var, g in zip(dis.trainable_variables, dis_grad):
                            tf.summary.histogram(
                                f"{var.name}/depth_{current_depth}_gradients",
                                g, step=step,
                                )

                    gen_grad, gen_loss = gen_apply_grad(
                        gen, dis,
                        loss_fn, gan_input,
                        tf.constant(current_depth, tf.int32),
                        tf.constant(alpha, tf.float32),
                        )
                    with self.tf_board_writer.as_default():
                        for var, g in zip(gen.trainable_variables, gen_grad):
                            tf.summary.histogram(
                                f"{var.name}/depth_{current_depth}_gradients",
                                g, step=step,
                                )

                    # if self.use_ema:
                    #     self.ema.apply(gen.trainable_variables)

                    # provide a loss feedback
                    if (i % max(int(total_batches / max(feedback_factor, 1)), 1) == 0 or i == 1):
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s] batch: %d d_loss: %f g_loss: %f" % (elapsed, i, dis_loss, gen_loss))
                        with self.tf_board_writer.as_default():
                            tf.summary.scalar("dis_loss", dis_loss, step=step)
                            tf.summary.scalar("gen_loss", gen_loss, step=step)

                        # if self.use_ema:
                        #     update_average(feedback_generator, self.ema)
                        resolution_dir = self.train_log_dir / str(int(2 ** current_depth))
                        resolution_dir.mkdir(exist_ok=True)
                        create_grid(
                            samples=feedback_generator(fixed_input, current_depth, alpha),
                            img_file=resolution_dir / f"{epoch:05d}_{i:05d}_alpha_{alpha:.3f}.png",
                            )
                    # increment the alpha ticker
                    ticker += 1
                    step += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                if (epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[depth_list_index]):
                    gen.save_weights(self.model_dir / f"gen_depth_{current_depth}_epoch_{epoch}")
                    dis.save_weights(self.model_dir / f"dis_depth_{current_depth}_epoch_{epoch}")

                gc.collect()
                tf.keras.backend.clear_session()

    def weight_transfer(
        self,
        current_gen,
        current_dis,
        current_depth,
        epochs,
        latent_size,
        batch_sizes,
        real_images_to_downsample,
        ):

        prev_depth, prev_last_epoch, prev_batch = \
            current_depth - 1, epochs[current_depth - 1 - 2], batch_sizes[current_depth - 1 - 2]

        prev_gen = load_generator(self.model_dir, prev_depth, prev_last_epoch, latent_size, use_eql=True)
        prev_dis = load_discriminator(self.model_dir, prev_depth, prev_last_epoch, latent_size, self.depth, real_images_to_downsample, use_eql=True)

        prev_weights_gen = get_layer_weights(prev_gen)
        prev_weights_dis = get_layer_weights(prev_dis)

        # feed model input before loading the weights/checkpoints to properly initialize shapes
        current_batch = batch_sizes[current_depth - 1 - 2]
        current_gan_input = tf.convert_to_tensor(tf.random.normal([current_batch, latent_size]), dtype=tf.float32)
        current_downsampled_images = progressive_downsample_batch(self.depth, real_images_to_downsample, current_depth, 1.0)
        current_gen(current_gan_input, current_depth, 1.0)
        current_dis(current_downsampled_images, current_depth, 1.0)

        print("At current_depth", current_depth)
        print("current GENERATOR layers ......")
        current_gen = transfer(current_gen, prev_weights_gen)
        print("current DISCRIMINATOR layer ......")
        current_dis = transfer(current_dis, prev_weights_dis)
        del prev_gen, prev_dis, prev_weights_dis, prev_weights_gen

        return current_gen, current_dis
