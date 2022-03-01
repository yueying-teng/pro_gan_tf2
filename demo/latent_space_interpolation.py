import tensorflow as tf
import argparse
import cv2
from pathlib import Path
import numpy as np
from numpy.linalg import norm

import sys
sys.path.append("../")

from pro_gan.utils import adjust_dynamic_range, load_generator


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
        "--last_epoch",
        action="store",
        type=int,
        default=40,
        help="model weights trained at this epoch will be loaded",
        )

    parser.add_argument(
        "--latent_size",
        action="store",
        type=int,
        default=512,
        help="latent size for the generator",
        )

    parser.add_argument(
        "--video_length",
        action="store",
        type=int,
        default=20,
        help="length of video in seconds",
        )

    parser.add_argument(
        "--num_videos",
        action="store",
        type=int,
        default=3,
        help="number of output videos",
        )

    parser.add_argument(
        "--fps",
        action="store",
        type=int,
        default=60,
        help="Frames per second in the video",
        )

    parser.add_argument(
        "--depth",
        action="store",
        type=int,
        default=7,
        help="size of the generated images in the video: 2 ** depth",
        )

    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        default="./output",
        help="path to the output directory for the video",
        )

    args = parser.parse_args()

    return args


def slerp(val, low, high):
    # spherical linear interpolation (slerp)
    omega = np.arccos(np.clip(np.dot(low / norm(low), high / norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def interpolate_points(p1, p2, steps=10):
    # uniform interpolation between two points in latent space
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)

    return vectors


def create_video_writer(i, video_dir, depth):
    width =  int(2 ** depth)
    video_fn = f"{i:03d}_latent_space_interpolation_resolution_{width}.avi"
    video_fp = video_dir / video_fn
    video_writer = cv2.VideoWriter(str(video_fp), cv2.VideoWriter_fourcc((*"MJPG")), 60, (width, width))

    return video_writer, video_fp


def create_video(i, gen, depth, latent_size, inbetween_steps, video_dir):
    video_writer, video_fp = create_video_writer(i, video_dir, depth)

    point1 = np.random.randn(latent_size)
    point2 = np.random.randn(latent_size)
    interpolated = interpolate_points(point1, point2, inbetween_steps)

    for j in range(len(interpolated)):
        gan_input = tf.cast(np.expand_dims(interpolated[j], 0), tf.float32)
        img = gen(gan_input, depth, alpha=1.0)[0]
        # scale from [-1,1] to [0,1]
        img = adjust_dynamic_range(img, drange_in = (-1, 1), drange_out = (0, 1)) * 255
        img = np.array(img).astype(np.uint8)[:, :, ::-1]

        video_writer.write(img)

    print(f"saving video at {video_fp} ...")
    video_writer.release()


def main(args):
    inbetween_steps = int(args.video_length * args.fps)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "video"

    video_dir.mkdir(parents=True, exist_ok=True)

    # load the generator model
    gen = load_generator(model_dir, args.depth, args.last_epoch, args.latent_size, use_eql=True)

    for i in range(args.num_videos):
        create_video(i, gen, args.depth, args.latent_size, inbetween_steps, video_dir)

if __name__ == "__main__":
    main(parse_arguments())


"""
python3 latent_space_interpolation.py --model_dir "/work/notebooks/test_train/models/20220105-142028" --video_length 10 --depth 6
"""
