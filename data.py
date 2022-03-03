# 1. Load the dataset
# 2. Use the albumentations library to augment the dataset.

import os
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def augment_data(images, masks, save_path, augment=True):
    H = 256
    W = 256

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("\\")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("\\")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = CenterCrop(H, W, p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks =  [y, y1, y2, y3, y4, y5]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


def load_data(path,split=0.1):
    img_path = sorted(glob(os.path.join(path, "images\\*")))
    msk_path = sorted(glob(os.path.join(path, "masks\\*")))

    len_ids = len(img_path)

    train_size = int((80/100)*len_ids)
    valid_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for validation
    test_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for testing

    train_x, test_x = train_test_split(img_path, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(msk_path, test_size=test_size, random_state=42)

    train_x, valid_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




if __name__ == "__main__":
    """ Loading original images and masks. """
    np.random.seed(42)
    path = "C:\\Users\\PC\\Desktop\\CVC-ClinicDB\\"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path, split=0.1)

    create_dir("C:\\Users\\PC\\Desktop\\new_data\\train\\images\\")
    create_dir("C:\\Users\\PC\\Desktop\\new_data\\train\\masks\\")
    create_dir("C:\\Users\\PC\\Desktop\\new_data\\valid\\images\\")
    create_dir("C:\\Users\\PC\\Desktop\\new_data\\valid\\masks\\")
    create_dir("C:\\Users\\PC\\Desktop\\new_data\\test\\images\\")
    create_dir("C:\\Users\\PC\\Desktop\\new_data\\test\\masks\\")

    augment_data(train_x, train_y, "C:\\Users\\PC\\Desktop\\new_data\\train\\", augment=True)
    augment_data(valid_x, valid_y, "C:\\Users\\PC\\Desktop\\new_data\\valid\\", augment=False)
    augment_data(test_x, test_y, "C:\\Users\\PC\\Desktop\\new_data\\test\\", augment=False)


