import os
import re
import pydicom
import numpy as np

from typing import Tuple
from PIL import Image

from utils.util import sorted_dcm

def _dwi_classification(intensity_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Image type classification for diffusion-weighted image.

    Classifies slices of diffusion-weighted images into b0 or b1000 by its voxel intensity.

    Args:
        intensity_array (np.ndarray): Numpy array containing slice index, vertical postion of slice and voxel intensity.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing array of b0/b1000 slice index.

    """
    # b0/b1000 seperation
    b0_test = np.zeros((0), dtype=np.int)
    bp_test = np.zeros((0), dtype=np.int)

    for arr in intensity_array:
        idx = intensity_array[:,1] == arr[1]
        intensity_value = intensity_array[idx,2]
        intensity_loc = intensity_array[idx,0]

        if np.sum(idx) == 1:
            bp_test = np.append(bp_test,intensity_loc[0])
        elif np.sum(idx) == 2:
            for rep in range(2):
                if intensity_value[rep] == np.min(intensity_value):
                    bp_test = np.append(bp_test,intensity_loc[rep])
                else:
                    b0_test = np.append(b0_test,intensity_loc[rep])
        else:
            raise Exception('More than 3 types of images detected. Please check the DICOM file.')

    b0_test = np.unique(b0_test).astype(np.int)
    bp_test = np.unique(bp_test).astype(np.int)

    return (b0_test, bp_test)

def read_dicom_dwi(case_path: str,
                   classify_b1000: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """DICOM file reader.

    Reads DICOM files from path and returns 3D b1000 (and b0 if available) images.

    Args:
        case_path (str): Path for DICOM folder where DICOM files are located.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing b0 and b1000 images.

    Usage:
        from JBS.read_dicom_dwi import read_dicom_dwi

        mri_path = r'mri\'
        case = 'DUM_0002'

        b1000_path = os.path.join(mri_path,case,'dwi','dcm','b1000')
        b0_img, b1000_img = read_dicom_dwi(b1000_path) # b0_img : empty array

    """
    flist = sorted_dcm(case_path)
    files_number = [re.sub('\D',' ',fname) for fname in flist]
    files_number = [fname.strip() for fname in files_number]
    files_number = [int(fname.split(' ')[-1]) for fname in files_number]
    files_number = np.unique(files_number)

    if len(files_number) != len(flist):
        flist = [fname for fname in flist if len(fname.lower().split('dwi')[-1]) > 8]

    intensity_array = []
    info_dwi_sliceloc = np.zeros(len(flist))
    start_dcm = 0

    for idx, fname in enumerate(flist):
        dcm_path = os.path.join(case_path, fname)
        dcm_info = pydicom.dcmread(dcm_path,force=True)
        if not hasattr(dcm_info.file_meta,'TransferSyntaxUID'):
            dcm_info.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        if not hasattr(dcm_info, 'PixelRepresentation'):
            dcm_info.PixelRepresentation = 0
        if not hasattr(dcm_info, 'PhotometricInterpretation'):
            dcm_info.PhotometricInterpretation = 'MONOCHROME2'
        if not hasattr(dcm_info, 'BitsStored'):
            dcm_info.BitsStored = 12
        image = dcm_info.pixel_array

        if start_dcm == 0:
            dwi = np.zeros((image.shape[0],image.shape[1],len(flist)), dtype=image[0,0].dtype)
            dwi[:,:,idx] = image
            start_dcm = 1
        else:
            dwi[:,:,idx] = image

        if hasattr(dcm_info,'ImagePositionPatient'):
            intensity = np.mean(image[image>30])
            intensity_array = intensity_array + [np.array([idx,dcm_info.ImagePositionPatient[2],intensity])]
            info_dwi_sliceloc[idx] = dcm_info.ImagePositionPatient[2]

    intensity_array = np.array(intensity_array)
    if classify_b1000:
        [b0_test,bp_test] = _dwi_classification(intensity_array)
    else:
        b0_test, bp_test = range(dwi.shape[-1]), range(dwi.shape[-1])

    dwi_b0 = dwi[:, :, b0_test]
    dwi_bp = dwi[:, :, bp_test]

    info_dwi_sliceloc = info_dwi_sliceloc[info_dwi_sliceloc!=0]

    if info_dwi_sliceloc[-1] - info_dwi_sliceloc[0] < 0:
        dwi_b0 = np.flip(dwi_b0, 2)
        dwi_bp = np.flip(dwi_bp, 2)

    return (dwi_b0, dwi_bp)

def read_gt(
    dcm_path: str,
    gt_path: str) -> np.ndarray:
    # get sorted gt file names
    dcm_list = sorted_dcm(dcm_path)
    gt_list = [fname.replace('.dcm','.png') for fname in dcm_list]

    # stack gt
    dcm_info = pydicom.dcmread(os.path.join(dcm_path,dcm_list[0]), force=True, stop_before_pixels=True)
    row = dcm_info.Rows
    col = dcm_info.Columns

    gt = np.zeros((row,col,len(gt_list))).astype(bool)
    for idx, fname in enumerate(gt_list):
        gt_fpath = os.path.join(gt_path, fname)
        gt_slice = Image.open(gt_fpath)
        gt_slice = np.array(gt_slice)[:,:,0]
        gt_slice[gt_slice != 0 ] = 1
        gt[:,:,idx] = gt_slice.astype(bool)
    
    return gt