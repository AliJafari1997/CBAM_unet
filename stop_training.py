import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from glob import glob
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from train import *
from model import build_model
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import CustomObjectScope
from utils import *
from metrics import *


if __name__=='__main__':

    np.random.seed(42)
    tf.random.set_seed(42)

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
    epochs = 50
    lr =1e-6

    metrics=[Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)




    callbacks = [
        ModelCheckpoint(model_path),
        CSVLogger("/content/drive/MyDrive/files/data.csv",append=True),
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]  


    train_steps=len(train_x)//batch_size
    valid_steps=len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model = load_model_weight("/content/drive/MyDrive/files/model.h5")


    model.compile(loss=dice_loss, optimizer=Nadam(lr), metrics=metrics)

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)
