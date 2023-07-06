import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import ExponentialCyclicalLearningRate
from tensorflow.keras.callbacks import ModelCheckpoint

from data.augmentation import (
    RandomAffineV2,
    RandomBiasField,
    RandomFlip,
    RandomGamma
)
from data.data_loader import data_loader
from model.model import UNet_encoder, UNet_decoder
from model.loss import focal_tversky_loss, dice_coef

parser = argparse.ArgumentParser(description="DWI segmentation implementation")

#########################
#### data parameters ####
#########################
parser.add_argument("--list_path", default="", type=str, help="path to load data list")
parser.add_argument("--data_path", default="", type=str, help="path to dataset repository")
parser.add_argument("--img_size", default=[256,256], nargs="+", help="row and column size of image slice")
parser.add_argument("--n_slices", default=64, type=int, help="maximum number of slices for 3D image")
parser.add_argument("--n_channels", default=1, type=int, help="image channel size")
parser.add_argument("--scale_range", default=[0.9,1.1], nargs="+", help="rescaling range for random affine")
parser.add_argument("--rot_range", default=[-90,90], nargs="+", help="rotation angle range for random affine")
parser.add_argument("--trans_range", default=[-5,5], nargs="+", help="translation range for random affine")
parser.add_argument("--order", default=2, type=int, help="order of polynomial function for random bias field")
parser.add_argument("--coeff_range", default=[-0.5,0.5], nargs="+", help="coefficient range for random bias field")
parser.add_argument("--flip_axes", default=[None,0,1,2], nargs="+", help="axes for random flip")
parser.add_argument("--log_gamma", default=[-0.3,0.3], nargs="+", help="coefficient range for random gamma")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=1000, type=int, help="number of total epochs to run")
parser.add_argument("--batch_size", default=2, type=int, help="dataloader batch size")
parser.add_argument("--optim_param", default=[0.6,0.4,1.33], nargs="+", help="optimizer parameters for Focal Tversky")
parser.add_argument("--lr_range", default=[1e-5,1e-4], nargs="+", help="range of cyclical learning rate")
parser.add_argument("--da_model_fpath", default="", type=str,
                    help="File path to baseline model to be used in domain adaptation. \
                          If given, changes to domain adaptation training.")

#########################
#### other parameters ###
#########################
parser.add_argument("--gpu_id", default=0, type=int, help='Device id of GPU to use. Falls back to CPU mode if -1 is given.')
parser.add_argument("--ckpt_path", default="", type=str, help='path to save checkpoints')
parser.add_argument("--nb_filters", default=[12,24,48,96,192], nargs="+", help="number of filters for the model")
parser.add_argument("--seed", type=int, default=31, help="seed")

def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

    if args.gpu_id != -1:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        print(len(gpu_devices))
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    train_list = np.load(os.path.join(args.list_path, 'train_list.npy'))
    val_list = np.load(os.path.join(args.list_path, 'val_list.npy'))

    # Build dataloader
    rand_affine = RandomAffineV2(
        scale_range=tuple(args.scale_range),
        rot_range=tuple(args.rot_range),
        trans_range=tuple(args.trans_range)
    )
    rand_bias = RandomBiasField(order=args.order, coeff_range=tuple(args.coeff_range))
    rand_flip = RandomFlip(flip_axes=args.flip_axes)
    rand_gamma = RandomGamma(log_gamma=args.log_gamma)
    transform_list = [rand_affine, rand_bias, rand_flip, rand_gamma]

    train_loader = data_loader(
        train_list, args.data_path, batch_size=args.batch_size,
        img_size=tuple(args.img_size), n_slices=args.n_slices,
        n_channels=args.n_channels, normalize='avg', transform=transform_list,
        shuffle=True, repeat=True, prefetch=True
    )
    val_loader = data_loader(
        val_list, args.data_path, batch_size=args.batch_size,
        img_size=tuple(args.img_size), n_slices=args.n_slices,
        n_channels=args.n_channels, normalize='avg', transform=None,
        shuffle=False, repeat=False, prefetch=True
    )
    print("Building data done with {} and {} images loaded.".format(len(train_list), len(val_list)))

    # Build model
    K.clear_session()

    img_in = Input(shape=(*args.img_size, args.n_slices, args.n_channels), name='img_in')
    encoder_out = UNet_encoder(img_in, nb_filter=args.nb_filters)
    model_out = UNet_decoder(encoder_out, nb_filter=args.nb_filters)
    model = Model(inputs=img_in, outputs=model_out)

    step_size = round(len(train_list) / args.batch_size)
    lr = ExponentialCyclicalLearningRate(
        initial_learning_rate=args.lr_range[0],
        maximal_learning_rate=args.lr_range[1],
        step_size=step_size
        )
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    loss = focal_tversky_loss(
        alpha=args.optim_param[0],
        beta=args.optim_param[1],
        gamma=args.optim_param[2],
        smooth=1e-7
    )

    model.compile(optimizer=opt, loss=loss, metrics=[dice_coef])
    print("Building model done.")

    # Model training
    if len(args.da_model_fpath) != 0:
        print("")
        print("Model path given for domain adaptation. Changing settings...")
        for idx, layer in enumerate(model.layers):
            if 'batch_normalization' in layer.name:
                continue
            elif idx <= 34:
                layer.trainable = False
            else:
                continue
        
        model.load_weights(args.da_model_fpath)
        model.compile(optimizer=opt, loss=loss, metrics=[dice_coef])
        print("Domain adaptation setup done.")
        print("Baseline model: {}".format(os.path.basename(args.da_model_fpath)))
        print("")
    
    callbacks = []
    if len(args.ckpt_path) != 0:
        print("Checkpoint path given. Saving model weights from now on.")
        print("")
        if not os.path.isdir(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        ckpt = ModelCheckpoint(
            os.path.join(args.ckpt_path,'ckpt_epoch-{epoch:04d}-{val_loss:.4f}.h5'),
            monitor='val_loss', mode='min', save_weights_only=True,
            save_best_only=True)
        callbacks.append(ckpt)
    
    print('Start training...')
    model.fit(
        train_loader,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=step_size,
        validation_data=val_loader
    )

if __name__ == "__main__":
    main()