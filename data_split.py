import argparse
import os

import numpy as np

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Image preparation for DWI segmentation")

parser.add_argument("--data_path", default="", type=str, help="path to preprocessed images")
parser.add_argument("--save_path", default="", type=str, help="path to save data lists")
parser.add_argument("--cnt_list", default=[], nargs="+",
                    help="List of medical center codenames for train-test stratification. \
                          This should contain codenames for all centers, or be left blank.")
parser.add_argument("--val_size", default=0.2, type=float, help="ratio for dataset size for each validation dataset")
parser.add_argument("--test_size", default=0.2, type=float, help="ratio of dataset size for Internal test dataset")

def main():
    global args
    args = parser.parse_args()

    rt_path = os.path.join(args.data_path, 'rt_img')
    flist = os.listdir(rt_path)

    np.save(os.path.join(args.save_path, 'full_dataset.npy', flist))

    # Train-and-validation / Internal test dataset split
    train_val_list = []
    internal_test_list = []

    if len(args.cnt_list) != 0:
        for cnt in args.cnt_list:
            list_temp = [fname for fname in flist if cnt in fname]
            train_list_temp, test_list_temp = train_test_split(list_temp, test_size=args.test_size)
            train_val_list.extend(train_list_temp)
            internal_test_list.extend(test_list_temp)
    else:
        train_val_list, internal_test_list = train_test_split(flist, test_size=args.test_size)
    
    np.save(os.path.join(args.save_path, 'train_val_list.npy'), train_val_list)
    np.save(os.path.join(args.save_path ,'internal_test_list.npy'), internal_test_list)

    # Train / validation dataset split for each data size ratio
    size_list = [2.5, 5, 10, 20, 50, 100]
    size_name_list = ['2p5', '5', '10', '20', '50', '100']
    for size, size_name in zip(size_list, size_name_list):
        size_ratio = size / 100
        list_temp = train_val_list.copy()
        np.random.shuffle(list_temp)

        sub_list = list_temp[:round(len(list_temp) * size_ratio)]
        if len(sub_list) == 0 or len(sub_list) == 1:
            continue
        train_list, val_list = train_test_split(sub_list, test_size=args.val_size)

        if not os.path.isdir(os.path.join(args.save_path, size_name)):
            os.mkdir(os.path.join(args.save_path, size_name))
        np.save(os.path.join(args.save_path, size_name, 'train_list.npy'), train_list)
        np.save(os.path.join(args.save_path, size_name, 'val_list.npy'), val_list)

if __name__ == "__main__":
    main()