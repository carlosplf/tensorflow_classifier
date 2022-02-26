import pathlib
import tensorflow as tf


class MyDataLoader():
    def __init__(self, datapath="./dataset", batchsize=24):
        self.train_ds = None
        self.valid_ds = None
        self.classes = None
        self.datapath = datapath
        self.batchsize = batchsize

    def load(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.datapath + '/seg_train/seg_train',
            labels='inferred',
            label_mode='int',
            class_names=None,
            color_mode='rgb',
            batch_size=self.batchsize,
            image_size=(160,160),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        
        self.valid_ds = tf.keras.utils.image_dataset_from_directory(
            self.datapath + '/seg_test/seg_test',
            labels='inferred',
            label_mode='int',
            class_names=None,
            color_mode='rgb',
            batch_size=self.batchsize,
            image_size=(160,160),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False
        )

        self.classes = self.train_ds.class_names
