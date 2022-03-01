import tensorflow as tf
import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.append("../")

from pro_gan.utils import load_generator, create_grid


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        action="store",
        type=str,
        default="/work/notebooks/test_train/models/20220105-142028",
        help="path to the model checkpoints",
        required=True,
        )

    parser.add_argument(
        "--num_samples",
        action="store",
        type=int,
        default=25,
        help="number of images to tile the sheet",
        required=True,
        )

    parser.add_argument(
        "--num_sheets",
        action="store",
        type=int,
        default=3,
        help="number of output sheets",
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
        default=7,
        help="defines the size of the generated image: 2 ** depth",
        )

    parser.add_argument(
        "--last_epoch",
        action="store",
        type=int,
        default=40,
        help="model weights trained at this epoch will be loaded",
        )

    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        default="./output",
        help="path to the output directory for the sheet",
        )

    args = parser.parse_args()

    return args


def create_sheet(gen, num_samples, latent_size, depth):
    generated_samples = []
    gan_input = tf.random.normal([num_samples, latent_size])
    for j in range(len(gan_input)):
        img = gen(tf.cast(np.expand_dims(gan_input[j], 0), tf.float32), depth, alpha=1.0)[0]
        generated_samples.append(img)

    return generated_samples


def main(args):
    model_dir = Path(args.model_dir)

    output_dir = Path(args.output_dir)
    facesheets_dir = output_dir / "facesheets"
    facesheets_dir.mkdir(parents=True, exist_ok=True)

    gen = load_generator(model_dir, args.depth, args.last_epoch, args.latent_size, use_eql=True)

    for i in range(args.num_sheets):
        generated_samples = create_sheet(gen, args.num_samples, args.latent_size, args.depth)

        facesheets_fn = f"{i:03d}_facesheets_resolution_{int(2 ** args.depth)}.png"
        print(f"saving facesheet at {facesheets_dir / facesheets_fn} ...")
        create_grid(generated_samples, facesheets_dir / facesheets_fn, figure_size=(25, 25))


if __name__ == "__main__":
    main(parse_arguments())

"""
python3 create_facesheet.py --model_dir "/work/notebooks/test_train/models/20220105-142028" --num_samples 25 --depth 6
"""
