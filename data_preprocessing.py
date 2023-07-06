import argparse
import os
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from data.dicom import (
    read_dicom_dwi,
    read_gt
)
from data.preprocessing import (
    skull_stripping,
    n4_correction,
    trim_boundary,
    img_normalization,
    histogram_centering
)
from utils.util import resize_img_ax

parser = argparse.ArgumentParser(description="Image preparation for DWI segmentation")

parser.add_argument("--raw_path", default="", type=str, help="path to DICOM and GT files.")
parser.add_argument("--save_path", default="", type=str, help="path to save processed images")
parser.add_argument("--n_workers", default=1, type=int,
                    help="Number of workers for preprocessing image. \
                          The maximum of this number will be limited to number of CPU cores.")

def main():
    global args
    args = parser.parse_args()

    # Set n_jobs
    if args.n_workers == 0:
        args.n_workers = 1
    cpu_count = multiprocessing.cpu_count()
    
    n_jobs = min(args.n_workers, cpu_count)
    print("Using parallel processing...")
    print("Number of workers for image preprocessing: {}".format(n_jobs))

    case_list = os.listdir(args.raw_path)
    print("Number of cases to be preprocessed: {}".format(len(case_list)))

    def preprocess_data(case):
        dcm_path = os.path.join(args.raw_path, case, 'dcm')
        gt_path = os.path.join(args.raw_path, case, 'gt')

        _, img = read_dicom_dwi(dcm_path)
        ss_img = skull_stripping(img.astype(np.float64))
        n4_img = n4_correction(ss_img)
        trim_img = trim_boundary(n4_img)
        rt_img = trim_img
        rt_img = img_normalization(rt_img, clipping=False)
        rt_img = histogram_centering(rt_img)
        rt_img = rt_img.clip(min=0, max=255)
        rt_img = rt_img.round(0)
        rt_img = rt_img.astype(np.uint8)

        gt = read_gt(dcm_path, gt_path)

        x, y, _ = rt_img.shape
        if (x, y) != (256, 256):
            rt_img = resize_img_ax(rt_img, (256, 256), order=3)
            gt = resize_img_ax(gt, (256, 256), order=0)
            rt_img = rt_img.clip(min=0, max=255)
            rt_img = rt_img.round(0)
            rt_img = rt_img.astype(np.uint8)
        
        np.save(os.path.join(
            args.save_path,
            'rt_img',
            '{}.npy'.format(case)
        ), rt_img)
        np.save(os.path.join(
            args.save_path,
            'gt',
            '{}.npy'.format(case)
        ), gt)
    
    if not os.path.isdir(os.path.join(args.save_path, 'rt_img')):
        os.makedirs(os.path.join(args.save_path, 'rt_img'))
    if not os.path.isdir(os.path.join(args.save_path, 'gt')):
        os.makedirs(os.path.join(args.save_path, 'gt'))

    _ = Parallel(n_jobs=n_jobs, verbose=1)(delayed(preprocess_data)(case) for case in case_list)

if __name__ == "__main__":
    main()