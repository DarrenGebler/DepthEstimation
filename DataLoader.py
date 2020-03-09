from glob import glob
import Helpers
import numpy as np


class DataLoader:
    def __init__(self, dataset_name="pix2pix-depth"):
        self.dataset_name = dataset_name
        base_path = "./input/" + self.dataset_name + "/" + self.dataset_name + "/"
        self.training_path = base_path + "training/"
        self.validation_path = base_path + "validation/"
        self.testing_path = base_path + "testing/"
        self.testing_raw_path = base_path + "testing_raw/"

    def load_random_data(self, data_size, is_testing=False):
        paths = glob(self.training_path+"*") if is_testing else glob(self.testing_path+"*")
        source_images, destination_images = Helpers.Helpers.image_pairs(np.random.choice(paths, size=data_size), is_testing)
        return Helpers.Helpers.normalise(source_images), Helpers.Helpers.normalise(destination_images)

    def yield_batch(self, batch_size, is_testing=False):
        paths = glob(self.training_path+"*") if is_testing else glob(self.validation_path+"*")
        for i in range(int(len(paths)/batch_size)-1):
            batch = paths[i*batch_size:(i+1)*batch_size]
            source_images, destination_images = Helpers.Helpers.image_pairs(batch, is_testing)
            yield Helpers.Helpers.normalise(source_images), Helpers.Helpers.normalise(destination_images)
