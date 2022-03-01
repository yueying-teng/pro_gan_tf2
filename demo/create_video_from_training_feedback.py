import tensorflow as tf
import argparse
import os
import cv2
from pathlib import Path
import numpy as np

import sys
sys.path.append("../")



def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        action="store",
        type=str,
        default="/work/notebooks/test_train/logs/20220105-142028",
        help="path to the saved images during training",
        required=True,
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
        help="size of the images used to generate the video: 2 ** depth",
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


def main(args):
    log_dir = Path(f"{args.log_dir}/{int(2 ** args.depth)}")

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "video"
    video_fn = f"combined_training_feedback_resolution_{int(2 ** args.depth)}.avi"
    video_fp = video_dir / video_fn

    video_dir.mkdir(parents=True, exist_ok=True)

    temp_iter = Path(log_dir).iterdir()
    height, width, _ =  cv2.imread(str(log_dir / next(temp_iter))).shape
    video_writer = cv2.VideoWriter(str(video_fp), cv2.VideoWriter_fourcc((*"MJPG")), args.fps, (width, height))

    for item in sorted(Path(log_dir).iterdir(), key=os.path.getmtime):
        if item.is_file():
            img = cv2.imread(str(log_dir / item))
            video_writer.write(img)

    print(f"saving video at {video_fp} ...")
    video_writer.release()


if __name__ == "__main__":
    main(parse_arguments())

"""
python3 create_video_from_training_feedback.py --log_dir "/work/notebooks/test_train/logs/20220105-142028" --depth 7
"""
