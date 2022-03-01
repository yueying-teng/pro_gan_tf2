# %%
import argparse
from pathlib import Path
import sys
sys.path.append("../")

from pro_gan.gan import ProGANTrainer
from pro_gan.losses import StandardGAN, WganGP, LSGAN

import tensorflow as tf
tf.config.run_functions_eagerly(True)

# %%
def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        default="./test_train",
        help="path to save the training logs and model checkpoints",
        )

    parser.add_argument(
        "--data_dir",
        action="store",
        type=str,
        default="/work/CELEBAHQ/train",
        help="path to the training data",
        required=True,
        )

    parser.add_argument(
        "--latent_size",
        action="store",
        type=int,
        default=512,
        help="latent size for the generator",
        )

    parser.add_argument(
        "--depth",
        action="store",
        type=int,
        default=10,
        help="defines the resolution, 2 ** depth, of the generated images once training is done",
        )

    parser.add_argument(
        "--dis_learning_rate",
        action="store",
        type=float,
        default=0.001,
        help="discriminator training learning rate",
        )

    parser.add_argument(
        "--gen_learning_rate",
        action="store",
        type=float,
        default=0.001,
        help="generator training learning rate",
        )

    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        default=40,
        help="training epochs for each depth",
        )

    parser.add_argument(
        "--crop_size",
        action="store",
        type=int,
        default=None,
        help="crop training data to square images",
        )

    parser.add_argument(
        "--start_depth",
        action="store",
        type=int,
        default=2,
        help="start training from models with resolution 2 ** start_depth",
        )

    parser.add_argument(
        "--fade_in_percentage",
        action="store",
        type=int,
        default=50,
        help="percentage of epochs per resolution to use fade-in \
            (for fading in the new layers; not used for 4 x 4 models, \
            but a dummy value is still needed)",
        )

    args = parser.parse_args()

    return args


def main(args):
    batch_sizes = [256, 256, 128, 64, 16, 6, 3, 2, 1][:args.depth - 1]

    progan_trainer = ProGANTrainer(
        depth=args.depth,
        latent_size=args.latent_size,
        use_ema=False,
        save_dir=Path(args.save_dir),
        gen_learning_rate=args.gen_learning_rate,
        dis_learning_rate=args.dis_learning_rate,
        )

    with tf.device("/gpu:0"):
        progan_trainer.train(
            data_dir=Path(args.data_dir),
            epochs=[args.epochs] * (args.depth - 1),
            fade_in_percentages=[args.fade_in_percentage] * (args.depth - 1),
            batch_sizes=batch_sizes,
            crop_size=args.crop_size,
            start_depth=args.start_depth,
            loss_fn=WganGP(),
            feedback_factor=20,
            checkpoint_factor=20,
            )

if __name__ == "__main__":
    main(parse_arguments())

"""
python3 train_pro_gan.py --data_dir "/work/CELEBAHQ/train" --epochs 10
"""
