import numpy as np
import os
import random
import tensorflow as tf
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    """
    Args:
        resize_to: int or None, if not None, resize the original images
        to (resize_to, resize_to, #channels)
        crop_size: target size of the cropped images,
                   output dimension: (crop_size, crop_size, channels)

    Returns:
        gather all image filepaths in the given directory and return
        [-1, 1] normalized image arrays with label string (optional)
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        mirror_flip=True,
        crop_size=None,
        resize_to=None,
        shuffle=True,
        ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_paths, self.labels = self.gather_subfolder_img_path(data_dir)
        self.indices = [i for i in range(len(self.labels))]
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.mirror_flip = mirror_flip
        self.on_epoch_end()

    def __len__(self):
        # number of batched per epoch
        return int(len(self.labels) // float(self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_img = []
        batch_img_path = [
            self.img_paths[i]
            for i in self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ]
        batch_labels = [
            self.labels[i]
            for i in self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ]

        for path in batch_img_path:
            img = cv2.imread(path)
            if img is None:
                continue
            if self.crop_size:
                img = self.crop(img)
            if self.resize_to is not None:
                img = cv2.resize(img, (self.resize_to, self.resize_to))
            batch_img.append(img)

        if self.mirror_flip:
            batch_img = self.apply_mirror_flip(batch_img)

        # convert to RGB images and normalize images to [-1, 1]
        batch_img = (np.array(batch_img)[...,::-1] - 127.5) / 127.5
        batch_img = tf.cast(batch_img, tf.float32)
        return batch_img
        # return batch_img, batch_labels

    def apply_mirror_flip(self, batch_img):
        mask = np.random.rand(len(batch_img)) < 0.5
        batch_img = np.array(batch_img)
        batch_img[mask] = batch_img[mask, :, ::-1]
        return batch_img

    def crop(self, img):
        # crop rectangle images to square images by retaining the shorter side
        height, width, _ = img.shape
        if height >= width:
            assert height >= self.crop_size, (
                f"{self.crop_size} is greater than long side of the image"
            )
            y = random.randint(0, height - self.crop_size)
            return img[y:self.crop_size+ y, :, :]

        assert width >= self.crop_size, (
            f"{self.crop_size} is greater than the long side of the image"
        )
        x = random.randint(0, width - self.crop_size)
        return img[ :, x:self.crop_size + x, :]

    def gather_subfolder_img_path(self, data_dir):
        paths = []
        labels = []
        for root, _, fnames in sorted(os.walk(data_dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if self.is_valid_file(path):
                    paths.append(path)
        for path in paths:
            labels.append(str(path.split('/')[-2]))
        return paths, labels

    @staticmethod
    def is_valid_file(path):
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return os.path.splitext(path)[-1] in img_extensions


def load_dataset(data_dir, batch_size, crop_size, mirror_flip=False, resize_to=None):
    """
    create a DataGenerator using the given arguments

    Args:
        data_dir: dir of the dataset
        batch_size: batch size for training
        crop_size: num of parallel readers for reading the data

    Returns: dataloader for the dataset
    """

    return DataGenerator(
        crop_size=crop_size,
        resize_to=resize_to,
        data_dir=data_dir,
        mirror_flip=mirror_flip,
        batch_size=batch_size,
        shuffle=True,
    )
