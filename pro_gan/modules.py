import tensorflow as tf
import numpy as np

from pro_gan.custom_layers import (
    PixelNormalization,
    MinibatchStdDev,
    EqualizedDense,
    EqualizedConv2D,
    )

from tensorflow.keras.layers import (
    AveragePooling2D,
    UpSampling2D,
    Flatten,
    Dense,
    Reshape,
    Conv2D,
    Embedding,
    LeakyReLU,
    )


class GenInitialBlock(tf.keras.layers.Layer):
    """
    Initial conv block of the generator
    use pixelwise normalization of the feature vectors after each
    Conv 3 × 3 layer in the generator
                        output shape
    Latent vector       1 x 1 x 512
    Conv 4 x 4          4 x 4 x 512
    Conv 3 × 3          4 x 4 x 512

    Args:
        use_eql: whether to use equalized learning rate
        latent_size: dimension of the input latent vector, e.g. 512
        out_channels: number of output channels
    """

    def __init__(self, latent_size, out_channels, use_eql, name=None):
        super(GenInitialBlock, self).__init__(name=name)
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.use_eql = use_eql

        DenseBlock = EqualizedDense if use_eql else Dense
        ConvBlock = EqualizedConv2D if use_eql else Conv2D

        if use_eql:
            self.dense = DenseBlock(units=4 * 4 * out_channels, gain=np.sqrt(2) / 4, name="gen_initblock_dense")
        else:
            self.dense = DenseBlock(units=4 * 4 * out_channels, name="gen_initblock_dense")
        self.reshape = Reshape((4, 4, out_channels))
        self.conv_1 = ConvBlock(filters=out_channels, kernel_size=(4, 4), padding="SAME", name="gen_initblock_conv1")
        self.conv_2 = ConvBlock(filters=out_channels, kernel_size=(3, 3), padding="SAME", name="gen_initblock_conv2")
        self.pixel_norm = PixelNormalization()
        self.lrelu = LeakyReLU(0.2)

    def call(self, x):
        y = self.pixel_norm(x)   # normalize the latents to hypersphere
        y = self.dense(y)
        y = self.reshape(y)
        y = self.pixel_norm(self.lrelu(y))
        y = self.lrelu(self.conv_1(y))
        y = self.pixel_norm(y)
        y = self.lrelu(self.conv_2(y))
        y = self.pixel_norm(y)
        return y

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(self.latent_size))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class GenGeneralConvBlock(tf.keras.layers.Layer):
    """
    Module implementing a general conv block
                    output shape
    Conv 3 × 3         4 x 4 x 512
    Upsample           8 x 8 x 512 (*)
    Conv 3 × 3         8 x 8 x 512 (*)
    Conv 3 × 3         8 x 8 x 512 (*) the three marked layers make up
    a general conv block in the generator

    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels required
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, out_channels, use_eql, name=None):
        super(GenGeneralConvBlock, self).__init__(name=name)
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2D if use_eql else Conv2D

        self.upsample = UpSampling2D()
        self.conv_1 = ConvBlock(filters=out_channels, kernel_size=(3, 3), padding="SAME", name="gen_generalblock_conv1")
        self.conv_2 = ConvBlock(filters=out_channels, kernel_size=(3, 3), padding="SAME", name="gen_generalblock_conv2")
        self.pixel_norm = PixelNormalization()
        self.lrelu = LeakyReLU(0.2)

    def call(self, x):
        y = self.upsample(x)
        y = self.pixel_norm(self.lrelu(self.conv_1(y)))
        y = self.pixel_norm(self.lrelu(self.conv_2(y)))

        return y


class DisFinalBlock(tf.keras.layers.Layer):
    """
    Final block for the Discriminator
                            output shape
    Downsample              4 x 4 x 512
    Minibatch stddev        4 x 4 x 512 (*)
    Conv 3 × 3              4 x 4 x 512 (*)
    Conv 4 × 4              1 x 1 x 512 (*)
    Fully-connected         1 x 1 x 512 (*) the marked four layers make
    up the final block of the discriminator

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels, out_channels, use_eql, name=None):
        super(DisFinalBlock, self).__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2D if use_eql else Conv2D
        DenseBlock = EqualizedDense if use_eql else Dense

        self.batch_discriminator = MinibatchStdDev()
        self.conv_1 = ConvBlock(filters=in_channels, kernel_size=(3, 3), padding="SAME", name="dis_finalblock_conv1")
        self.conv_2 = ConvBlock(filters=out_channels, kernel_size=(4, 4), padding="SAME", name="dis_finalblock_conv2")
        self.flatten = Flatten()
        self.dense = DenseBlock(units=1, gain=1, name="dis_finalblock_dense") \
            if self.use_eql else DenseBlock(units=1, name="dis_finalblock_dense")
        self.lrelu = LeakyReLU(0.2)

    def call(self, x):
        y = self.batch_discriminator(x)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.flatten(y)
        y = self.dense(y)
        return y


class DisGeneralConvBlock(tf.keras.layers.Layer):
    """
    General block in the discriminator
                        output shape
    Input image         1024 x 1024 x 3
    Conv 1 × 1          1024 x 1024 x 16
    Conv 3 × 3          1024 x 1024 x 16 (*)
    Conv 3 × 3          1024 x 1024 x 32 (*)
    Downsample          512 x 512 x 32   (*) the three marked layers
    make up a general block in the discriminator

    Args:
        in_channels: number of channels of the input tensor to the block
        out_channels: number of output channels of the block
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels, out_channels, use_eql, name=None):
        super(DisGeneralConvBlock, self).__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2D if use_eql else Conv2D

        self.conv_1 = ConvBlock(filters=in_channels, kernel_size=(3, 3), padding="SAME", name="dis_generalblock_conv1")
        self.conv_2 = ConvBlock(filters=out_channels, kernel_size=(3, 3), padding="SAME", name="dis_generalblock_conv2")
        self.downsample = AveragePooling2D()
        self.lrelu = LeakyReLU(0.2)

    def call(self, x):
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downsample(y)
        return y


# class ConDisFinalBlock(tf.keras.layers.Layer):
#     """
#     Final block for the Conditional Discriminator
#     Uses the Projection mechanism
#     from the paper -> https://arxiv.org/pdf/1802.05637.pdf

#     Args:
#         in_channels: number of input channels
#         num_classes: number of classes for conditional discrimination
#         use_eql: whether to use equalized learning rate
#     """

#     def __init__(self, out_channels, num_classes, use_eql):
#         super(ConDisFinalBlock, self).__init__()
#         self.out_channels = out_channels
#         self.num_classes = num_classes
#         self.use_eql = use_eql

#         ConvBlock = EqualizedConv2D if use_eql else Conv2D

#         self.conv_1 = ConvBlock(in_channels + 1, in_channels, (3, 3), padding="SAME")
#         self.conv_2 = ConvBlock(in_channels, out_channels, (4, 4))
#         self.conv_3 = ConvBlock(out_channels, 1, (1, 1))

#         # we also need an embedding matrix for the label vectors
#         self.label_embedder = Embedding(num_classes, out_channels, max_norm=1)
#         self.batch_discriminator = MinibatchStdDev()
#         self.lrelu = LeakyReLU(0.2)

#     def call(self, x, labels):
#         y = self.batch_discriminator(x)
#         y = self.lrelu(self.conv_1(y))
#         y = self.lrelu(self.conv_2(y))

#         # embed the labels
#         labels = self.label_embedder(labels)  # [B x C]

#         # compute the inner product with the label embeddings
#         y_ = torch.squeeze(torch.squeeze(y, dim=-1), dim=-1)  # [B x C]
#         projection_scores = (y_ * labels).sum(dim=-1)  # [B]

#         # normal discrimination score
#         y = self.lrelu(self.conv_3(y))  # This layer has linear activation

#         # calculate the total score
#         final_score = y.view(-1) + projection_scores

#         # return the output raw discriminator scores
#         return final_score
