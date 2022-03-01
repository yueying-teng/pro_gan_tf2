import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

import sys
sys.path.append("../")

from pro_gan.gan import ProGANTrainer
from pro_gan.losses import StandardGAN, WganGP, LSGAN
from pro_gan.utils import progressive_downsample_batch


def test_progressive_downsample_batch():
    batch = tf.random.normal([4, 1024, 1024, 3])
    batch = tf.clip_by_value(batch, clip_value_min=0, clip_value_max=1)
    total_depth = 10

    for res_log2 in range(2, 10):
        modified_batch = progressive_downsample_batch(
            total_depth, batch, depth=res_log2, alpha=0.001
            )
        print(f"Downsampled batch at depth {res_log2}: {modified_batch.shape}")
        plt.figure()
        plt.title(f"Image at resolution: {int(2 ** res_log2)}x{int(2 ** res_log2)}")
        plt.imshow(modified_batch[0])
        assert modified_batch.shape == (
            batch.shape[0],
            int(2 ** res_log2),
            int(2 ** res_log2),
            batch.shape[-1],
            )

    plt.figure()
    plt.title(f"Image at resolution: {1024}x{1024}")
    plt.imshow(batch[0])
    plt.show()


def test_pro_gan_train():
    depth = 5
    latent_size = 128
    dis_learning_rate = 0.001
    gen_learning_rate = 0.001
    data_dir = Path("/work/CELEBAHQ/train")

    progan_trainer = ProGANTrainer(
        depth=depth,
        latent_size=latent_size,
        use_ema=False,
        save_dir=Path("./test_train"),
        gen_learning_rate=gen_learning_rate,
        dis_learning_rate=dis_learning_rate,
        )

    progan_trainer.train(
        data_dir=data_dir,
        epochs=[20 for _ in range(depth - 1)],
        fade_in_percentages=[50 for _ in range(depth - 1)],
        batch_sizes=[64, 64, 64, 32],
        crop_size=None,
        start_depth=2,
        loss_fn=WganGP(),
        feedback_factor=20,
        checkpoint_factor=20,
        )
    print("test_finished")


def main():
    test_pro_gan_train()


if __name__ == "__main__":
    main()
