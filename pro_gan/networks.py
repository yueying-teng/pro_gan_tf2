import numpy as np
import tensorflow as tf

from pro_gan.custom_layers import EqualizedConv2D
from pro_gan.modules import (
    # ConDisFinalBlock,
    DisFinalBlock,
    DisGeneralConvBlock,
    GenGeneralConvBlock,
    GenInitialBlock,
    )
from tensorflow.keras.layers import (
    AveragePooling2D,
    UpSampling2D,
    Conv2D,
    Activation,
    )


def nf(
    stage: int,
    fmap_base: int = 16 << 10,
    fmap_decay: float = 1.0,
    fmap_min: int = 1,
    fmap_max: int = 512,
    ):
    """
    computes the number of fmaps present in each stage, i.e. ouput depth
    e.g.
    stage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nf[stage] = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]

    Args:
        stage: stage level
        fmap_base: base number of fmaps
        fmap_decay: decay rate for the fmaps in the network
        fmap_min: minimum number of fmaps
        fmap_max: maximum number of fmaps

    Returns: number of fmaps that should be present there
    """
    return int(np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max,))


class Generator(tf.keras.Model):
    """
    Generator Module (block) of the GAN network
    training starts from resolution 4 x 4

    Args:
        depth: required depth of the Network
        num_channels: number of output channels (default = 3 for RGB)
        latent_size: size of the latent manifold
        use_eql: whether to use equalized learning rate
    """

    def __init__(
        self,
        depth = 10,
        num_channels = 3,
        latent_size = 512,
        use_eql = True,
        ):
        super(Generator, self).__init__()

        self.depth = depth
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2D if use_eql else Conv2D

        self.g_layers = [GenInitialBlock(latent_size, nf(1), use_eql, name="GenInitialBlock")]

        for stage in range(1, depth - 1):
            self.g_layers.append(GenGeneralConvBlock(nf(stage + 1), use_eql, name=f"GenGeneralConvBlock{stage}"))

        if depth == 2:
            self.to_rgb = [
                ConvBlock(
                    filters=num_channels,
                    kernel_size=(1, 1),
                    gain=1,
                    activation=Activation("tanh"),
                    name="to_rgb_conv1",
                    )
                if self.use_eql else
                ConvBlock(
                    filters=num_channels,
                    kernel_size=(1, 1),
                    activation=Activation("tanh"),
                    name="to_rgb_conv1",
                    )
                ]
        else:
            self.to_rgb = [
                ConvBlock(
                    filters=num_channels,
                    kernel_size=(1, 1),
                    gain=1,
                    activation=Activation("tanh"),
                    name=f"to_rgb_conv{stage}",
                    )
                if self.use_eql else
                ConvBlock(
                    filters=num_channels,
                    kernel_size=(1, 1),
                    activation=Activation("tanh"),
                    name=f"to_rgb_conv{stage}",
                    )
                for stage in range(depth - 2, depth)
                ]

    def call(self, x, depth, alpha):
        """
        Args:
            x: input latent noise
            depth: depth from where the network's output is required
            alpha: value of alpha for fade-in effect

        Returns: generated images at the give depth's resolution

        e.g. depth = 4
            g_layer = [
                GenInitialBlock: nf(1),
                GenGeneralConvBlock1: nf(2),
                GenGeneralConvBlock2: nf(3),
                ]
            to_rgb = [to_rgb2, to_rgb3]

        - GenInitialBlock: nf(1)
        - GenGeneralConvBlock1: nf(2)
        - residual:		straight:
          to_rgb2	    GenGeneralConvBlock1: nf(3)
          Upsample 		ro_rgb3
        - Combine residual & straight
        """

        assert depth <= self.depth, f"Requested output depth {depth} cannot be produced"

        if depth == 2:
            y = self.to_rgb[0](self.g_layers[0](x))
        else:
            y = x
            for layer_block in self.g_layers[: depth - 2]:
                y = layer_block(y)
            residual = UpSampling2D()(self.to_rgb[0](y))
            straight = self.to_rgb[1](self.g_layers[depth - 2](y))
            y = (alpha * straight) + ((1 - alpha) * residual)
        return y

    def get_config(self):
        return {
            "depth": self.depth,
            "num_channels": self.num_channels,
            "latent_size": self.latent_size,
            "use_eql": self.use_eql,
            }


class Discriminator(tf.keras.Model):
    """
    Discriminator of the GAN

    Args:
        depth: depth of the discriminator. log_2(resolution)
        num_channels: number of channels of the input images (Default = 3 for RGB)
        latent_size: latent size of the final layer
        use_eql: whether to use the equalized learning rate
        num_classes: number of classes for a conditional discriminator (Default = None)
                     meaning unconditional discriminator
    """

    def __init__(
        self,
        depth = 7,
        num_channels = 3,
        latent_size = 512,
        use_eql = True,
        num_classes = None,
        ):
        super(Discriminator, self).__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.use_eql = use_eql
        self.num_classes = num_classes
        self.conditional = num_classes is not None

        ConvBlock = EqualizedConv2D if use_eql else Conv2D

        if self.conditional:
            # self.d_layers = [ConDisFinalBlock(nf(1), latent_size, num_classes, use_eql)]
            pass
        else:
            self.d_layers = [DisFinalBlock(nf(1), latent_size, use_eql, name="DisFinalBlock")]

        for stage in range(1, depth - 1):
            self.d_layers.append(
                DisGeneralConvBlock(nf(stage + 1), nf(stage), use_eql, name=f"DisGeneralConvBlock{stage}")
                )

        if depth == 2:
            self.from_rgb = [
                ConvBlock(
                    filters=nf(1),
                    kernel_size=(1, 1),
                    activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                    name=f"from_rgb_conv1",
                    )
                ]
        else:
            self.from_rgb = [
                ConvBlock(
                    filters=nf(stage),
                    kernel_size=(1, 1),
                    activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                    name=f"from_rgb_conv{stage}",
                    )
                for stage in range(depth - 2, depth)
                ]

    def call(self, x, depth, alpha, labels= None):
        """
        forward pass of the discriminator

        Args:
            x: input to the network
            depth: current depth of operation (Progressive GAN)
            alpha: current value of alpha for fade-in
            labels: labels for conditional discriminator (Default = None)
                    shape => (Batch_size,) shouldn't be a column vector

        Returns: raw discriminator scores

        e.g. depth = 4
            d_layers = [
                DisFinalBlock: nf(1),
                DisGeneralConvBlock2: in=nf(2), out=nf(1),
                DisGeneralConvBlock3: in=nf(3), out=nf(2),
                ]
            from_rgb = [from_rgb2: nf(2), from_rgb3: nf(3)]

        - residual:				straight:
          Downsample	    	from_rgb3: nf(3)
          from_rgb2: nf(2) 		DisGeneralConvBlock3: in=nf(3), out=nf(2),
        - Combine residual & straight
        - DisGeneralConvBlock2: in=nf(2), out=nf(1),
        - DisFinalBlock: nf(1)
        """

        assert (depth <= self.depth), f"Requested output depth {depth} cannot be evaluated"

        if self.conditional:
            assert labels is not None, "Conditional discriminator requires labels"

        if depth > 2:
            residual = self.from_rgb[0](
                AveragePooling2D(pool_size=(2, 2), strides=2)(x)
            )
            straight = self.d_layers[(depth - 1 - 1)](self.from_rgb[1](x))
            y = (alpha * straight) + ((1 - alpha) * residual)

            for i in range(depth - 2 - 1, 0, -1):
                y = self.d_layers[i](y)
        else:
            y = self.from_rgb[0](x)
        if self.conditional:
            y = self.d_layers[0](y, labels)
        else:
            y = self.d_layers[0](y)
        return y

    def get_config(self):
        return {
            "depth": self.depth,
            "num_channels": self.num_channels,
            "latent_size": self.latent_size,
            "use_eql": self.use_eql,
            "num_classes": self.num_classes,
            }
