import os
import numpy as np
from glob import glob
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from sklearn.model_selection import train_test_split
from model import build_model
from utils import *
from metrics import *



def read_images(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_COLOR)
    x=cv2.resize(x,(256,256))
    x=x/255.0
    # (256,256,3)
    return x


def read_mask(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x=cv2.resize(x,(256,256))
    x=x/255.0
    x=np.expand_dims(x,axis=-1)
    # (256,256,1)
    return x

def tf_parse(x,y):
    def _parse(x,y):
        x=read_images(x)   # numpy array
        y=read_mask(y)     # numpy array
        return x,y
        # we must change their format from numpy array to tensor
    x,y=tf.numpy_function(_parse,[x,y],[tf.float64,tf.float64])
    x.set_shape([256,256,3])
    y.set_shape([256,256,1])

    return x,y

def tf_dataset(x,y,batch=8):
# x: list of image path, y: list of masks path
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset=dataset.map(tf_parse)  # read and give numpy array
    dataset=dataset.batch(batch)
    dataset=dataset.repeat()
    return dataset


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("/content/drive/MyDrive/files")

    train_path = "/content/drive/MyDrive/new_data/train/"
    valid_path = "/content/drive/MyDrive/new_data/valid/" 
    
    ## Training
    train_x = sorted(glob(os.path.join(train_path, "images/*")))
    train_y = sorted(glob(os.path.join(train_path, "masks/*")))

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Validation
    valid_x = sorted(glob(os.path.join(valid_path, "images/*")))
    valid_y = sorted(glob(os.path.join(valid_path, "masks/*")))

    model_path='/content/drive/MyDrive/files/model.h5'
    batch_size = 8
    epochs = 100
    lr = 1e-4

    input_shape = (256,256, 3)

    model = build_model(input_shape)

    #model=build_model()

    metrics=[Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    model.compile(loss=dice_loss, optimizer=Nadam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path),
        CSVLogger("/content/drive/MyDrive/files/data.csv"),
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]  


    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False) 
