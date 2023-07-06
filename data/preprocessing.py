from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from typing import Optional
import SimpleITK as sitk
import numpy as np

def skull_stripping(img: np.ndarray) -> np.ndarray:
    """Skull Stripping for DWI.

    Removes CSF and skull area from 3D diffusion-weighted image. Other modalities are not tested.

    Args:
        img (np.ndarray): 3D diffusion-weighted image stacked vertically.

    Returns:
        np.ndarray: 3D diffusion-weighted image with CSF and skull removed.

    """
    # Gaussian Filtering
    img_gauss = img.copy()
    img_gauss = gaussian(img_gauss, sigma=1, multichannel=True, preserve_range=True, truncate=4.0)

    # Otsu Thresholding
    otsu_val = threshold_otsu(img_gauss,nbins=256)
    otsu_label = img.copy()
    otsu_label[otsu_label < otsu_val] = 0
    otsu_label[otsu_label >= otsu_val] = 255

    # Slice 별로 일정 크기 이상의 object만 남기고 제거
    x, y, z = img.shape
    ratio = (x / 128) ** 2
    for slice_idx in range(otsu_label.shape[2]):
        otsu_label[:,:,slice_idx] = remove_small_objects(otsu_label[:,:,slice_idx].astype('bool'),
                                                         min_size=32 * ratio, connectivity=1)
        otsu_label[:,:,slice_idx] = remove_small_holes(otsu_label[:,:,slice_idx].astype('bool'),
                                                       area_threshold=10 * ratio, connectivity=1)

    # 원본 array에 mask 적용
    img_out = img.copy()
    img_out[otsu_label == 0] = 0

    return img_out

def n4_correction(img: np.ndarray,
                  num_iter: Optional[int] = 10) -> np.ndarray:
    """N4 bias correction for MRI

    Performs bias correction for MR images.

    Args:
        img (np.ndarray): 3D MR image stacked vertically. Float data type needed.
        num_iter (Optional[int]): Number of iterations to perform correction.

    Returns:
        np.ndarray: Bias corrected 3D MR image.

    """
    # np array to sitk
    sitk_img = sitk.GetImageFromArray(img)
    # mask image
    sitk_mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iter])

    img_out = corrector.Execute(sitk_img, sitk_mask)
    # sitk to np array
    img_out = sitk.GetArrayFromImage(img_out)

    return img_out

def histogram_centering(img: np.ndarray,
                        center_val: Optional[float] = 150) -> np.ndarray:
    """Pixel intensity centering.

    Adjusts voxel intensity to be centered around specified value.

    Args:
        img (np.ndarray): 3D MR image stacked vertically.
        center_val (Optional[float]): Center value of voxel intensity.

    Returns:
        np.ndarray: 3D MR image with adjusted voxel intensity.

    """
    img_out = img.copy()

    # Pixel 신호에 대한 histogram 작성
    data = img_out[img_out > 0].tolist()
    hist, centers = np.histogram(data, bins=256)

    # Histogram spline curve fitting
    centers = np.array([centers[0] + (np.diff(centers)[0] * i) for i in range(len(hist))])
    kde = gaussian_kde(data, bw_method='silverman')
    counts_spl = kde(centers)

    # Peak에 해당하는 histogram의 index 확인
    peak_list, _ = find_peaks(counts_spl, height=max(counts_spl) * 0.7, distance=1)

    # peak_list 중 center_val에 가장 가까운 histogram의 index 저장
    peak_distance = np.abs(center_val - centers[peak_list])
    min_idx = peak_distance.argmin()
    min_idx = peak_list[min_idx]

    # Image가 center_val에 centering되도록 coeff 설정 및 곱연산 수행
    coeff = center_val / centers[min_idx]
    img_out *= coeff

    return img_out

def trim_boundary(img: np.ndarray) -> np.ndarray:
    """Boundary noise removal.

    Removes any potential high-signal artifacts around image boundary.

    Args:
        img (np.ndarray): 3D MR image stacked vertically.

    Returns:
        np.ndarray: 3D MR image with boundary signal removed.

    """
    x, y, z = img.shape
    img_mask = np.zeros((x, y))
    img_mask[int(0.05 * x):int(0.95 * x),int(0.05 * y):int(0.95 * y)] = 1
    img_mask = img_mask.astype(np.bool)

    img_out = img.copy()
    for i in range(img_out.shape[2]):
        img_out[:,:,i][~img_mask] = 0

    return img_out

def img_normalization(img: np.ndarray,
                      clipping: Optional[bool] = True) -> np.ndarray:
    """Pixel intensity normalization.

    Normalize pixel intensity from 0~255 and converts data type to uint8.

    Args:
        img (np.ndarray): 3D MR image stacked vertically.
        clipping (Optional[bool]): If True, set image dtype to np.uint8.

    Returns:
        np.ndarray: 3D MR image in uint8 data type.

    """
    img_out = img.copy()

    np_max = np.max(img_out)
    if np.percentile(img_out, 99.5) + 50 < np_max:
        np_max = np.percentile(img_out, 99.5)

    img_out = 255 * (img_out / np_max)

    # Value clipping -> 0 ~ 255
    if clipping:
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out.round(0)
        img_out = img_out.astype(np.uint8)

    return img_out
