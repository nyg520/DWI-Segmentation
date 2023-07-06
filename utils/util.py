import os
import pydicom
import numpy as np

from typing import Tuple, Optional, List
from scipy.ndimage import zoom
from skimage.transform import resize


def resize_img(img: np.ndarray,
               target_shape: Tuple[int, ...],
               order: Optional[int] = None) -> np.ndarray:
    """Image resizing.

    Resize n-dimensional image to specified shape.

    Args:
        img (np.ndarray): N-dimensional image.
        target_shape (Tuple[int, ...]): Desired shape of image to resize.
        order (int): Order of interpolation. Must be in the range 0-5.
            * 0: Nearest-neighbor
            * 1: Bi-linear
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quadratic
            * 5: Bi-quintic
        Default is 0 if image.dtype is np.bool and 1 otherwise.

    Returns:
        np.ndarray: Resized n-dimensional image.

    """
    img_shape = img.shape
    reshape_ratio = [val1 / val2 for val1, val2 in zip(target_shape, img_shape)]

    if order == None:
        if img.dtype == bool:
            order = 0
        else:
            order = 1
    img_out = zoom(img, reshape_ratio, order=order)

    return img_out

def resize_img_ax(img: np.ndarray,
                  axial_target_shape: Tuple[int, int],
                  order: Optional[int] = None) -> np.ndarray:
    """Image resizing.

    Resize n-dimensional image to specified shape (slice by slice).
    Image shape should by (x,y,z), where z is slice axis.

    Args:
        img (np.ndarray): 3-dimensional image.
        target_shape (Tuple[int, ...]): Desired slice shape of image to resize.
        order (int): Order of interpolation. Must be in the range 0-5.
            * 0: Nearest-neighbor
            * 1: Bi-linear
            * 2: Bi-quadratic
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic
        Default is 0 if image.dtype is np.bool and 1 otherwise.

    Returns:
        np.ndarray: Resized 3-dimensional image.

    """
    if order == None:
        if img.dtype == bool:
            order = 0
        else:
            order = 1
    x, y, z = img.shape
    img_out = np.zeros((*axial_target_shape,z))
    for i in range(z):
        img_slice = img[:,:,i]
        img_slice = resize(img_slice, axial_target_shape, order=order, preserve_range=True)
        img_out[:,:,i] = img_slice.copy()
    img_out = img_out.astype(img.dtype)

    return img_out

def get_bbox(img_slice: np.ndarray) -> Tuple[int, ...]:
    """Bounding box for ROI in 2D image.

    Finds bounding box for voxel values bigger than 0 and returns its indices.

    Args:
        img_slice (np.ndarray): 2D slice of MR image. Skull stripping recommended.

    Returns:
        Tuple[int, ...]: Index of top/bottom/left/right point for the bounding box in the image.

    """
    top_bottom = img_slice.sum(axis=1)
    left_right = img_slice.sum(axis=0)

    top = np.where(top_bottom!=0)[0][0]
    bottom = np.where(top_bottom!=0)[0][-1]
    left = np.where(left_right!=0)[0][0]
    right = np.where(left_right!=0)[0][-1]

    return (top, bottom, left, right)

def sorted_dcm(dcm_path: str) -> List[str]:
    """DICOM file sorting.

    Sort DICOM files from single patient by its axial position.

    Args:
        dcm_path (str): Path to folder where DICOM files are located.

    Returns:
        List[str]: Sorted list of DICOM file names.

    """
    dcm_list = os.listdir(dcm_path)

    dcm_sorted = []
    for fname in dcm_list:
        dcm_info = pydicom.dcmread(os.path.join(dcm_path,fname), force=True, stop_before_pixels=True)
        pos_z = dcm_info.ImagePositionPatient[2]
        dcm_sorted.append((fname, float(pos_z)))

    dcm_sorted = sorted(dcm_sorted, key=lambda x : x[1])
    dcm_sorted = [fname for fname, pos_z in dcm_sorted]

    return dcm_sorted