# -*- coding: utf-8 -*-
import pickle
import cv2
import numpy as np
from keras.preprocessing.image import Iterator
import random
import os
import sys
import matplotlib.pyplot as plt


# %%
class BoxCarsDataset(object):
    # %%
    # change this to your location
    BOXCARS_DATASET_ROOT = "/home/cuky/Devel/data_link/BoxCars116k/"

    # %%
    BOXCARS_IMAGES_ROOT = os.path.join(BOXCARS_DATASET_ROOT, "images")
    BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl")
    BOXCARS_ATLAS = os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl")
    BOXCARS_CLASSIFICATION_SPLITS = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

    def __init__(self, load_atlas=False, load_split=None, use_estimated_3DBB=False, estimated_3DBB_path=None):
        self.dataset = load_cache(self.BOXCARS_DATASET)
        self.use_estimated_3DBB = use_estimated_3DBB

        self.atlas = None
        self.split = None
        self.split_name = None
        self.estimated_3DBB = None
        self.X = {}
        self.Y = {}
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None  # for labels as array of 0-1 flags

        if load_atlas:
            self.load_atlas()
        if load_split is not None:
            self.load_classification_split(load_split)
        if self.use_estimated_3DBB:
            self.estimated_3DBB = load_cache(estimated_3DBB_path)

        print("Datsaets Loaded:\nMain data %s\natlas %s" % (len(self.dataset), len(self.atlas)))
        print("##### Data loading finished:\n")

    # %%
    def load_atlas(self):
        self.atlas = load_cache(self.BOXCARS_ATLAS)

    # %%
    def load_classification_split(self, split_name):
        self.split = load_cache(self.BOXCARS_CLASSIFICATION_SPLITS)[split_name]
        self.split_name = split_name

    # %%
    def get_image(self, vehicle_id, instance_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)

    # %%
    def get_vehicle_instance_data(self, vehicle_id, instance_id, original_image_coordinates=False):
        """
        original_image_coordinates: the 3DBB coordinates are in the original image space
                                    to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                    which is done if this parameter is False.
        """
        vehicle = self.dataset["samples"][vehicle_id]
        instance = vehicle["instances"][instance_id]
        if not self.use_estimated_3DBB:
            bb3d = self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB"]
        else:
            bb3d = self.estimated_3DBB[vehicle_id][instance_id]

        if not original_image_coordinates:
            bb3d = bb3d - instance["3DBB_offset"]

        return vehicle, instance, bb3d


        # %%

    def initialize_data(self, part):
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized" % part
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label] * num_instances)
        self.X[part] = np.asarray(x, dtype=int)
        print("For-Loop finished\n")

        y = np.asarray(y, dtype=int)
        y_categorical = np.zeros((y.shape[0], self.get_number_of_classes()))
        y_categorical[np.arange(y.shape[0]), y] = 1
        self.Y[part] = y_categorical

        print("##### Data init finished:\n")

    def get_number_of_classes(self):
        return len(self.split["types_mapping"])

    def evaluate(self, probabilities, part="test", top_k=1):
        samples = self.X[part]
        assert samples.shape[0] == probabilities.shape[0]
        assert self.get_number_of_classes() == probabilities.shape[1]
        part_data = self.split[part]
        probs_inds = {}
        for vehicle_id, _ in part_data:
            probs_inds[vehicle_id] = np.zeros(len(self.dataset["samples"][vehicle_id]["instances"]), dtype=int)
        for i, (vehicle_id, instance_id) in enumerate(samples):
            probs_inds[vehicle_id][instance_id] = i

        get_hit = lambda probs, gt: int(gt in np.argsort(probs.flatten())[-top_k:])
        hits = []
        hits_tracks = []
        for vehicle_id, label in part_data:
            inds = probs_inds[vehicle_id]
            hits_tracks.append(get_hit(np.mean(probabilities[inds, :], axis=0), label))
            for ind in inds:
                hits.append(get_hit(probabilities[ind, :], label))

        return np.mean(hits), np.mean(hits_tracks)








#%%
class BoxCarsDataGenerator(Iterator):
    def __init__(self, dataset, part, batch_size=8, training_mode=True, seed=None, generate_y = True, image_size = (224,224)):
        assert image_size == (224,224), "only images 224x224 are supported by unpack_3DBB for now, if necessary it can be changed"
        assert dataset.X[part] is not None, "load some classification split first"
        super(BoxCarsDataGenerator, self).__init__(dataset.X[part].shape[0], batch_size, training_mode, seed)
        #Iterator.__init__(self, dataset.X[part].shape[0], batch_size, training_mode, seed)
        self.part = part
        self.generate_y = generate_y
        self.dataset = dataset
        self.image_size = image_size
        self.training_mode = training_mode
        if self.dataset.atlas is None:
            self.dataset.load_atlas()

        print("##### Data generator finished:\n")


    def next(self):
        print("next() called\n")
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        x = np.empty([current_batch_size] + list(self.image_size) + [3], dtype=np.float32)
        for i, ind in enumerate(index_array):
            print(len(index_array))
            vehicle_id, instance_id = self.dataset.X[self.part][ind]
            vehicle, instance, bb3d = self.dataset.get_vehicle_instance_data(vehicle_id, instance_id)
            image = self.dataset.get_image(vehicle_id, instance_id)
            print("In next() for-loop")
            plt.imshow(image)
            plt.show()
            if self.training_mode:
                print("---- Training mode True ----")
                image = alter_HSV(image) # randomly alternate color
                plt.imshow(image)
                plt.show()
                image = image_drop(image) # randomly remove part of the image
                plt.imshow(image)
                plt.show()
                bb_noise = np.clip(np.random.randn(2) * 1.5, -5, 5) # generate random bounding box movement
                flip = bool(random.getrandbits(1)) # random flip
                image, bb3d = add_bb_noise_flip(image, bb3d, flip, bb_noise)
                plt.imshow(image)
                plt.show()
            image = unpack_3DBB(image, bb3d)
            image = (image.astype(np.float32) - 116)/128.
            plt.imshow(image)
            plt.show()
            x[i, ...] = image
        if not self.generate_y:
            return x
        y = self.dataset.Y[self.part][index_array]
        return x, y








def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# %%
def alter_HSV(img, change_probability=0.6):
    if random.random() < 1 - change_probability:
        return img
    addToHue = random.randint(0, 179)
    addToSaturation = random.gauss(60, 20)
    addToValue = random.randint(-50, 50)
    hsvVersion = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    channels = hsvVersion.transpose(2, 0, 1)
    channels[0] = ((channels[0].astype(int) + addToHue) % 180).astype(np.uint8)
    channels[1] = (np.maximum(0, np.minimum(255, (channels[1].astype(int) + addToSaturation)))).astype(np.uint8)
    channels[2] = (np.maximum(0, np.minimum(255, (channels[2].astype(int) + addToValue)))).astype(np.uint8)
    hsvVersion = channels.transpose(1, 2, 0)

    return cv2.cvtColor(hsvVersion, cv2.COLOR_HSV2RGB)


# %%
def image_drop(img, change_probability=0.6):
    if random.random() < 1 - change_probability:
        return img
    width = random.randint(int(img.shape[1] * 0.10), int(img.shape[1] * 0.3))
    height = random.randint(int(img.shape[0] * 0.10), int(img.shape[0] * 0.3))
    x = random.randint(int(img.shape[1] * 0.10), img.shape[1] - width - int(img.shape[1] * 0.10))
    y = random.randint(int(img.shape[0] * 0.10), img.shape[0] - height - int(img.shape[0] * 0.10))
    img[y:y + height, x:x + width, :] = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    return img


# %%
def add_bb_noise_flip(image, bb3d, flip, bb_noise):
    bb3d = bb3d + bb_noise
    if flip:
        bb3d[:, 0] = image.shape[1] - bb3d[:, 0]
        image = cv2.flip(image, 1)
    return image, bb3d


# %%
def _unpack_side(img, origPoints, targetSize):
    origPoints = np.array(origPoints).reshape(-1, 1, 2)
    targetPoints = np.array([(0, 0), (targetSize[0], 0), (0, targetSize[1]),
                             (targetSize[0], targetSize[1])]).reshape(-1, 1, 2).astype(origPoints.dtype)
    m, _ = cv2.findHomography(origPoints, targetPoints, 0)
    resultImage = cv2.warpPerspective(img, m, targetSize)
    return resultImage


# %%
def unpack_3DBB(img, bb):
    frontal = _unpack_side(img, [bb[0], bb[1], bb[4], bb[5]], (75, 124))
    side = _unpack_side(img, [bb[1], bb[2], bb[5], bb[6]], (149, 124))
    roof = _unpack_side(img, [bb[0], bb[3], bb[1], bb[2]], (149, 100))

    final = np.zeros((224, 224, 3), dtype=frontal.dtype)
    final[100:, 0:75] = frontal
    final[0:100, 75:] = roof
    final[100:, 75:] = side

    return final

batch_size=8
print("##### Starting data loading:\n")
dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
print("##### Starting data init:\n")
dataset.initialize_data("train")
print("##### Starting data generator:\n")
generator_train = BoxCarsDataGenerator(dataset, "train", batch_size, training_mode=True)
x, y = generator_train.next()

