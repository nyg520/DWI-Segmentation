from typing import List, Optional, Tuple
import tensorflow as tf
import numpy as np
import os

from data.augmentation import rand_aug
from data.normalize import (
    avg_standardization,
    minmax_normalization,
    mode_standardization
)

def data_loader(list_IDs: List[str], load_path: str, batch_size: Optional[int] = 2,
               img_size: Optional[Tuple[int,int]] = (256,256),
               n_slices: Optional[int] = 64, n_channels: Optional[int] = 1,
               normalize: Optional[str] = 'avg', transform: Optional[bool] = None,
               shuffle: Optional[bool] = False, repeat: Optional[bool] = False,
               prefetch: Optional[bool] = True):
    # Data generator
    data_loader = DataLoader(list_IDs, load_path, img_size=img_size, n_slices=n_slices,
                             n_channels=n_channels, normalize=normalize, transform=transform)

    # tf.data List loader
    list_idx = list(range(len(list_IDs)))
    tf_generator = tf.data.Dataset.from_generator(lambda: list_idx, tf.uint32)

    # List loader -> Data generator (mapping)
    tf_generator = tf_generator.apply(tf.data.experimental.assert_cardinality(len(list_IDs)))
    if shuffle:
        tf_generator = tf_generator.shuffle(buffer_size=len(list_IDs), reshuffle_each_iteration=True)
        tf_generator = tf_generator.repeat()
    tf_generator = tf_generator.map(lambda i: tf.py_function(func=data_loader, inp=[i],
                                                             Tout=[tf.float32, tf.float32]),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tf_generator = tf_generator.batch(batch_size)
    if prefetch:
        tf_generator = tf_generator.prefetch(tf.data.experimental.AUTOTUNE)

    return tf_generator

def DataLoader(list_IDs, load_path, img_size=(256,256), n_slices=64,
               n_channels=1, normalize='avg', transform=None):
    def DataProcessor(i):
        i = i.numpy()
        case_ID = list_IDs[i]

        # initialization
        X = np.zeros((*img_size, n_slices, n_channels), dtype=np.float32)
        y = np.zeros((*img_size, n_slices, n_channels), dtype=np.float32)

        X_temp = np.load(os.path.join(load_path,'rt_img',case_ID)).astype(np.float32)
        y_temp = np.load(os.path.join(load_path,'gt',case_ID)).astype(np.float32)

        if transform != None:
            X_temp, y_temp = rand_aug(X_temp, y_temp, transform)

        # Image Standardization
        if normalize == 'avg':
            normalizer = avg_standardization
        elif normalize == 'minmax':
            normalizer = minmax_normalization
        elif normalize == 'mode':
            normalizer = mode_standardization
        X_temp = normalizer(X_temp)

        # Image Size Fix
        z_len = X_temp.shape[2]
        start = int((n_slices - z_len) / 2)
        X[:,:,start:start+z_len,0] = X_temp
        y[:,:,start:start+z_len,0] = y_temp

        return (X, y)

    return DataProcessor