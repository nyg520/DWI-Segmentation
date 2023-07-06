import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from data.normalize import avg_standardization

from model.model import UNet_encoder, UNet_decoder
from model.metric import dice_coef, sensitivity, precision

from utils.segmentation import dwi_segmentation, binarization

parser = argparse.ArgumentParser(description="DWI segmentation evaluation")

#########################
#### data parameters ####
#########################
parser.add_argument("--list_fpath", default="", type=str, help="file path to numpy array containing case ids for evaluation")
parser.add_argument("--data_path", default="", type=str, help="path to dataset repository")
parser.add_argument("--img_size", default=[256,256], nargs="+", help="row and column size of image slice")
parser.add_argument("--n_slices", default=64, type=int, help="maximum number of slices for 3D image")
parser.add_argument("--n_channels", default=1, type=int, help="image channel size")
parser.add_argument("--csv_fpath", default="", type=str, help="path for csv file to save evaluation results")

#########################
#### other parameters ###
#########################
parser.add_argument("--gpu_id", default=0, type=int, help='Device id of GPU to use. Falls back to CPU mode if -1 is given.')
parser.add_argument("--model_fpath", default="", type=str, help='file path to model checkpoint for evaluation')
parser.add_argument("--nb_filters", default=[12,24,48,96,192], nargs="+", help="number of filters for the model")
parser.add_argument("--round_digit", default=3, type=int, help="")

def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

    if args.gpu_id != -1:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    
    # Load file name list
    test_list = np.load(args.list_fpath)
    print("Using {} images for model evaluation.".format(len(test_list)))

    # Load model & checkpoint
    K.clear_session()

    img_in = Input(shape=(*args.img_size, args.n_slices, args.n_channels), name='img_in')
    encoder_out = UNet_encoder(img_in, nb_filter=args.nb_filters)
    model_out = UNet_decoder(encoder_out, nb_filter=args.nb_filters)
    model = Model(inputs=img_in, outputs=model_out)

    model.load_weights(args.model_fpath)
    print("Building model done.")
    print("Model loaded for evaluation: {}".format(os.path.basename(args.model_fpath)))

    dsc_list = []
    stv_list = []
    prc_list = []
    for fname in tqdm(test_list, total=len(test_list)):
        rt_img = np.load(os.path.join(args.data_path, 'rt_img', fname))
        y_true = np.load(os.path.join(args.data_path, 'gt', fname))

        img_in = avg_standardization(rt_img)
        y_pred = dwi_segmentation(img_in, model, clear_session=False)
        y_pred = binarization(y_pred)

        dsc = dice_coef(y_true, y_pred)
        stv = sensitivity(y_true, y_pred)
        prc = precision(y_true, y_pred)

        dsc_list.append(dsc)
        stv_list.append(stv)
        prc_list.append(prc)
    
    df = pd.DataFrame(data={
        'image_no': [case.replace('.npy','') for case in test_list],
        'dice_coeff':dsc_list,
        'sensitivity':stv_list,
        'precision':prc_list
    })
    df = df.set_index('image_no')
    df = df.round(args.round_digit)
    base_path = Path(args.csv_fpath).parent
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    df.to_csv(args.csv_fpath)

if __name__ == "__main__":
    main()