import scipy.stats as stats
import numpy as np

def avg_standardization(rt_img: np.ndarray) -> np.ndarray:
    """Image standardization for model input.

    Standardizes 3D image with mean and standard deviation of its voxel intensity bigger than zero.

    Args:
        rt_img (np.ndarray): 2D/3D MR image.

    Returns:
        np.ndarray: 2D/3D MR image with voxel intensities having zero mean and unit variance.

    """
    rt_out = rt_img.copy()
    rt_out = rt_out.astype(np.float32)

    # Set image roi
    roi = rt_out[rt_out > 0]

    # roi average & std
    avg = roi.mean()
    std = roi.std()

    # Image standardization
    rt_out = (rt_out - avg) / std

    return rt_out

def mode_standardization(rt_img: np.ndarray) -> np.ndarray:
    """Image mode standardization for model input.

    Standardizes 3D image with mode and standard deviation of its voxel intensity.
    Mode(most frequent value) will be selected from voxel quantile range [0.1,0.95], while
    standard deviation is caculated from voxels bigger than zero.

    Args:
        rt_img (np.ndarray): 2D/3D MR image.

    Returns:
        np.ndarray: Standardized 2D/3D MR image.

    """
    rt_out = rt_img.copy()
    rt_out = rt_out.astype(np.float32)

    # Set image roi
    roi = rt_out[rt_out > 0]

    # Quantile range
    q_low = np.quantile(roi, 0.1)
    q_high = np.quantile(roi, 0.95)
    q_range = roi[(roi > q_low) & (roi < q_high)]

    # quantile range mode & roi std
    mode, _ = stats.mode(q_range)
    mode = mode.item()
    std = roi.std()

    # Image standardization
    rt_out = (rt_out - float(mode)) / std

    return rt_out

def minmax_normalization(rt_img: np.ndarray) -> np.ndarray:
    """Image min-maxing for model input.

    Performs 3D image normalization with min-max scaling.

    Args:
        rt_img (np.ndarray): 2D/3D MR image.

    Returns:
        np.ndarray: Normalized 2D/3D MR image.

    """
    rt_out = rt_img.copy()
    rt_out = rt_out.astype(np.float32)

    rt_out -= np.min(rt_out)
    rt_out /= np.max(rt_out)
    rt_out -= 0.5
    rt_out *= 2

    return rt_out