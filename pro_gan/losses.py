import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


class StandardGAN():
    def __init__(self):
        self.criterion = BinaryCrossentropy(from_logits=True)

    def dis_loss(
        self,
        discriminator,
        real_samples,
        fake_samples,
        depth,
        alpha,
        labels = None,
        ):
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            real_scores = discriminator(real_samples, depth, alpha, labels, training=True)
            fake_scores = discriminator(fake_samples, depth, alpha, labels, training=True)
        else:
            real_scores = discriminator(real_samples, depth, alpha, training=True)
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)

        real_loss = self.criterion(tf.ones_like(real_scores), real_scores)
        fake_loss = self.criterion(tf.zeros_like(fake_scores), fake_scores)

        return (real_loss + fake_loss) / 2

    def gen_loss(
        self,
        discriminator,
        fake_samples,
        depth,
        alpha,
        labels = None,
        ):
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels, training=True)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)

        return self.criterion(tf.ones_like(fake_scores), fake_scores)


class LSGAN():
    def dis_loss(
        self,
        discriminator,
        real_samples,
        fake_samples,
        depth,
        alpha,
        labels = None,
        ):
        """
        https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L775
        """

        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            # TODO: implement conditional lsgan loss
            pass
        else:
            real_scores = discriminator(real_samples, depth, alpha, training=True)
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)
            real_loss = tf.reduce_mean(
                tf.math.squared_difference(tf.ones_like(real_scores), real_scores))
            fake_loss = tf.reduce_mean(
                tf.math.squared_difference(tf.zeros_like(fake_scores), fake_scores))

        return (real_loss + fake_loss) / 2

    def gen_loss(
        self,
        discriminator,
        fake_samples,
        depth,
        alpha,
        labels = None,
        ):
        """
        https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L775
        """

        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels, training=True)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)

        return tf.reduce_mean(tf.math.squared_difference(tf.ones_like(fake_scores), fake_scores))


class WganGP():
    """
    Wgan-GP loss function.
    Discriminator is required for computing the gradient penalty.

    Args:
        drift: weight for the drift penalty
    """

    def __init__(self, drift = 0.001):
        self.drift = drift

    def gradient_penalty(
        self,
        discriminator,
        real_samples,
        fake_samples,
        depth,
        alpha,
        labels,
        reg_lambda=10,
        ):
        """
        Args:
            discriminator: the discriminator used for computing the penalty
            real_samples: real samples
            fake_samples: fake samples
            depth: current depth in the optimization
            alpha: current alpha for fade-in
            reg_lambda: regularization lambda

        Returns: computed gradient penalty
        """

        # generate random epsilon
        alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_samples - real_samples
        # create the merge of both real and fake samples
        merged = real_samples + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(merged)
            # forward pass
            if labels is not None:
                assert discriminator.conditional, "labels passed to an unconditional discriminator"
                merged_pred = discriminator(merged, depth, alpha, labels, training=True)
            else:
                merged_pred = discriminator(merged, depth, alpha, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(merged_pred, [merged])[0]
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1.0e-8)
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
        penalty = reg_lambda * gradient_penalty

        return penalty

    def dis_loss(
        self,
        discriminator,
        real_samples,
        fake_samples,
        depth,
        alpha,
        labels=None,
        ):
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            real_scores = discriminator(real_samples, depth, alpha, labels, training=True)
            fake_scores = discriminator(fake_samples, depth, alpha, labels, training=True)
        else:
            real_scores = discriminator(real_samples, depth, alpha, training=True)
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)

        loss =  tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores) + \
                self.drift * tf.reduce_mean(tf.square(real_scores))
        # calculate the WGAN-GP (gradient penalty)
        loss += self.gradient_penalty(
            discriminator, real_samples, fake_samples, depth, alpha, labels,
            )

        return loss

    def gen_loss(
        self,
        discriminator,
        fake_samples,
        depth,
        alpha,
        labels = None,
        ):
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels, training=True)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha, training=True)

        return -tf.reduce_mean(fake_scores)
