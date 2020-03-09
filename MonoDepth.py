# Architecture taken directly from pixp2pix (https://phillipi.github.io/pix2pix)
import Helpers
import DataLoader
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob
import imageio
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class MonoDepth:
    def __init__(self):
        Helpers.Helpers.new_dir("./output/training/")
        Helpers.Helpers.new_dir("./output/training/losses/")

        self.image_shape = (256, 256, 3)
        self.data_loader = DataLoader.DataLoader()

        patch = int(256 / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.generator_filters = 64
        self.discriminator_filters = 64

        optimsiser = tf.keras.optimizers.Adam(0.00015, 0.5)
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss="mse", optimizer=optimsiser, metrics=["accuracy"])
        self.generator = self.generator()

        source_image = tf.keras.layers.Input(shape=self.image_shape)
        destination_image = tf.keras.layers.Input(shape=self.image_shape)
        generated_image = self.generator(destination_image)

        self.discriminator.trainable = False
        valid = self.discriminator([generated_image, destination_image])
        self.combined = tf.keras.models.Model(inputs=[source_image, destination_image],
                                              outputs=[valid, generated_image])
        self.combined.compile(loss=["mse", "mae"], loss_weights=[1, 100], optimizer=optimsiser)

    def generator(self):
        def conv2d(layer_input, filters, bn=True):
            downsample = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same")(layer_input)
            downsample = tf.keras.layers.LeakyReLU(alpha=0.2)(downsample)
            if bn:
                downsample = tf.keras.layers.BatchNormalization(momentum=0.8)(downsample)
            return downsample

        def deconv2d(layer_input, skip_input, filters, dropout_rate=0):
            upsample = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            upsample = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=1, padding="same", activation="relu")(
                upsample)
            if dropout_rate:
                upsample = tf.keras.layers.Dropout(dropout_rate)(upsample)
            upsample = tf.keras.layers.BatchNormalization(momentum=0.8)(upsample)
            upsample = tf.keras.layers.Concatenate()([upsample, skip_input])
            return upsample

        downsample_0 = tf.keras.layers.Input(shape=self.image_shape)
        downsample_1 = conv2d(downsample_0, self.generator_filters, bn=False)
        downsample_2 = conv2d(downsample_1, self.generator_filters * 2)
        downsample_3 = conv2d(downsample_2, self.generator_filters * 4)
        downsample_4 = conv2d(downsample_3, self.generator_filters * 8)
        downsample_5 = conv2d(downsample_4, self.generator_filters * 8)
        downsample_6 = conv2d(downsample_5, self.generator_filters * 8)
        downsample_7 = conv2d(downsample_6, self.generator_filters * 8)

        upsample_0 = deconv2d(downsample_7, downsample_6, self.generator_filters * 8)
        upsample_1 = deconv2d(upsample_0, downsample_5, self.generator_filters * 8)
        upsample_2 = deconv2d(upsample_1, downsample_4, self.generator_filters * 8)
        upsample_3 = deconv2d(upsample_2, downsample_3, self.generator_filters * 4)
        upsample_4 = deconv2d(upsample_3, downsample_2, self.generator_filters * 2)
        upsample_5 = deconv2d(upsample_4, downsample_1, self.generator_filters)
        upsample_6 = tf.keras.layers.UpSampling2D(size=2)(upsample_5)

        output_image = tf.keras.layers.Conv2D(3, kernel_size=4, strides=1, padding="same", activation="tanh")(
            upsample_6)
        return tf.keras.models.Model(downsample_0, output_image)

    def discriminator(self):
        def discriminator_layer(layer_input, filters, bn=True):
            discriminator_layer = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same")(layer_input)
            discriminator_layer = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator_layer)
            if bn:
                discriminator_layer = tf.keras.layers.BatchNormalization(momentum=0.8)(discriminator_layer)
            return discriminator_layer

        source_image = tf.keras.layers.Input(shape=self.image_shape)
        destination_image = tf.keras.layers.Input(shape=self.image_shape)
        combined_images = tf.keras.layers.Concatenate(axis=-1)([source_image, destination_image])
        discriminator_layer_1 = discriminator_layer(combined_images, self.discriminator_filters, bn=False)
        discriminator_layer_2 = discriminator_layer(discriminator_layer_1, self.discriminator_filters * 2)
        discriminator_layer_3 = discriminator_layer(discriminator_layer_2, self.discriminator_filters * 4)
        discriminator_layer_4 = discriminator_layer(discriminator_layer_3, self.discriminator_filters * 8)
        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="same")(discriminator_layer_4)
        return tf.keras.models.Model([source_image, destination_image], validity)

    def preview_training_progress(self, epoch, size=3):
        def preview_outputs(epoch, size):
            source_images, destination_images = self.data_loader.load_random_data(size, is_testing=True)
            generated_images = self.generator.predict(destination_images)
            grid_image = None
            for i in range(size):
                row = Helpers.Helpers.unnormalise(
                    np.concatenate([destination_images[i], generated_images[i], source_images[i]], axis=1))
                if grid_image is None:
                    grid_image = row
                else:
                    grid_image = np.concatenate([grid_image, row], axis=0)
            plt.imshow(grid_image / 255)
            plt.show()
            plt.close()
            grid_image = cv2.imwrite("./output/training/ " + str(epoch) + ".png", grid_image)

        def preview_losses():
            def plot(title, data):
                plt.plot(data, alpha=0.6)
                plt.title(title + "_" + str(i))
                plt.savefig("./output/training/losses/" + title + "_" + str(i) + ".png")
                plt.close()

            for i, d in enumerate(self.d_losses):
                plot("discriminator", d)
            for i, g in enumerate(self.g_losses):
                plot("generator", g)

        preview_outputs(epoch, size)

    def train(self):
        valid = np.ones((32,) + self.disc_patch)
        fake = np.zeros((32,) + self.disc_patch)
        self.d_losses = []
        self.g_losses = []
        self.preview_training_progress(0)
        for epoch in range(50):
            epoch_d_losses = []
            epoch_g_losses = []
            for iteration, (source_images, destination_images) in enumerate(self.data_loader.yield_batch(32)):
                generated_images = self.generator.predict(destination_images)
                d_loss_real = self.discriminator.train_on_batch([source_images, destination_images], valid)
                d_loss_fake = self.discriminator.train_on_batch([generated_images, destination_images], fake)
                d_losses = 0.5 * np.add(d_loss_real, d_loss_fake)
                g_losses = self.combined.train_on_batch([source_images, destination_images], [valid, source_images])
                epoch_d_losses.append(d_losses)
                epoch_g_losses.append(g_losses)
                print("\repoch:  " + str(epoch) + ", iteration: " + str(iteration) + ", d_losses: " + str(
                    d_losses) + ", g_losses: " + str(g_losses), sep=" ", end=" ", flush=True)
                self.d_losses.append(np.average(epoch_d_losses, axis=0))
                self.g_losses.append(np.average(epoch_g_losses, axis=0))
                self.preview_training_progress(epoch)

    def test(self):
        image_paths = glob(self.data_loader.testing_raw_path + "*")
        for image_path in image_paths:
            image = np.array(imageio.imread(image_path))
            image_normalised = Helpers.Helpers.normalise(image)
            generated_batch = self.generator.predict(np.array([image_normalised]))
            concat = Helpers.Helpers.unnormalise(np.concatenate([image_normalised, generated_batch[0]], axis=1))
            cv2.imwrite(
                ("./output/" + os.path.basename(image_path), cv2.cvtColor(np.float32(concat), cv2.COLOR_RGB2BGR)))


if __name__ == '__main__':
    output = MonoDepth()
    output.train()
    output.test()
