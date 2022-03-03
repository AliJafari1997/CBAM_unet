import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from data import load_data
from train import tf_dataset
from metrics import *
from tensorflow.keras.metrics import *
from model import build_model
from metrics import *
from utils import *


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("/content/drive/MyDrive/results/")
    ## Dataset
    test_path = '/content/drive/MyDrive/new_data/test/'
    batch_size = 8

    test_x = sorted(glob(os.path.join(test_path, "images/*")))
    test_y = sorted(glob(os.path.join(test_path, "masks/*")))
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model = load_model_weight("/content/drive/MyDrive/files/model.h5")


    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['acc',Recall(),Precision(),iou])

    model.evaluate(test_dataset, steps=test_steps)


    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"/content/drive/MyDrive/results/{i}.jpg", image)
