import numpy as np
import cv2
import imageio
import shutil
import os


class Helpers:
    @staticmethod
    def normalise(images):
        return np.array(images)/127.5-1

    @staticmethod
    def unnormalise(images):
        return (0.5*np.array(images)+0.5)*255

    @staticmethod
    def resize(image, size):
        return np.array(cv2.resize(image, size))

    @staticmethod
    def split_images(image, is_testing):
        image = imageio.imread(image).astype(np.float)
        _, width, _ = image.shape
        half_width = int(width/2)
        source_image = image[:, half_width:, :]
        destination_image = image[:, :half_width, :]
        source_image = Helpers.resize(source_image, (256, 256))
        destination_image = Helpers.resize(destination_image, (256, 256))

        if not is_testing and np.random.random() > 0.5:
            source_image = np.fliplr(source_image)
            destination_image = np.fliplr(destination_image)

        return source_image, destination_image

    @staticmethod
    def new_dir(path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def archive_output():
        shutil.make_archive("output", "zip", "./output")

    @staticmethod
    def image_pairs(batch, is_testing):
        source_images, destination_images = [], []
        for image_path in batch:
            source_image, destination_image = Helpers.split_images(image_path, is_testing)
            source_images.append(source_image)
            destination_images.append(destination_image)
        return source_images, destination_images

