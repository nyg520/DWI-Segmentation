from skimage.transform import AffineTransform, warp, rotate
from typing import List, Tuple
import numpy as np
import SimpleITK as sitk
import random

def get_interpolation(method):
    if method == 'nn':
        interpolator = sitk.sitkNearestNeighbor
    elif method == 'linear':
        interpolator = sitk.sitkLinear
    elif method == 'bspline':
        interpolator = sitk.sitkBSpline
    elif method == 'lanczos':
        interpolator = sitk.sitkLanczosWindowedSinc
    else:
        raise ValueError('Interpolation method must be one of nn, linear, bspline or lanczos.')

    return interpolator

class RandomFlip:
    def __init__(self, flip_axes=[None,0,1,2]):
        self.flip_axes = flip_axes

    def get_params(self):
        return random.choice(self.flip_axes)

    def apply(self, rt_img, gt):
        axis = self.get_params()

        if axis == None:
            return rt_img, gt
        else:
            return (np.flip(rt_img, axis), np.flip(gt, axis))

class RandomGamma:
    def __init__(self, log_gamma=(-0.3,0.3)):
        self.log_gamma = log_gamma

    def get_params(self):
        gamma = np.random.uniform(*self.log_gamma)
        gamma = np.exp(gamma)

        return gamma

    def apply(self, rt_img, gt):
        gamma = self.get_params()

        rt_out = rt_img.astype(np.float32)
        rt_out = rt_out ** gamma

        return (rt_out, gt)

class RandomBiasField:
    def __init__(self, order=2, coeff_range=(-0.5,0.5)):
        self.order = order
        self.coeff_range = coeff_range

    def get_params(self):
        random_coeff = []
        for x_order in range(0, self.order + 1):
            for y_order in range(0, self.order + 1 - x_order):
                for _ in range(0, self.order + 1 - (x_order + y_order)):
                    number = np.random.uniform(*self.coeff_range)
                    random_coeff.append(number)

        return random_coeff

    def get_meshgrid(self, rt_img):
        x, y, z = rt_img.shape
        shape = np.array([x, y, z])
        half_z = shape / 2

        ranges = [np.arange(-n, n) + 0.5 for n in half_z]

        meshes = np.asarray(np.meshgrid(*ranges))

        for mesh in meshes:
            mesh_max = mesh.max()
            if mesh_max > 0:
                mesh /= mesh_max

        x_mesh, y_mesh, z_mesh = meshes

        return x_mesh, y_mesh, z_mesh

    def get_bfield(self, rt_img):
        bias_field = np.zeros(rt_img.shape)

        random_coeff = self.get_params()
        x_mesh, y_mesh, z_mesh = self.get_meshgrid(rt_img)

        i = 0
        for x_order in range(self.order + 1):
            for y_order in range(self.order + 1 - x_order):
                for z_order in range(self.order + 1 - (x_order + y_order)):
                    coeff = random_coeff[i]
                    new_map = (coeff * x_mesh ** x_order * y_mesh ** y_order * z_mesh ** z_order)
                    bias_field += np.transpose(new_map, (1, 0, 2))
                    i += 1
        bias_field = np.exp(bias_field).astype(np.float32)

        return bias_field

    def apply(self, rt_img, gt):
        bias_field = self.get_bfield(rt_img)

        rt_out = rt_img.copy()
        rt_out = rt_out.astype(np.float32)
        rt_out = rt_out * bias_field

        return (rt_out, gt)

class RandomAffine:
    def __init__(self, scale_range=(0.9,1.1), rot_range=(-1,1), trans_range=(-5,5), interpolation='linear'):
        self.scale_range = scale_range
        self.rot_range = rot_range
        self.trans_range = trans_range
        self.interpolation = interpolation

    def get_params(self):
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale_z = np.random.uniform(*self.scale_range)
        scale = [scale_x, scale_y, scale_z]
        scale = np.array(scale, dtype=float)

        rotation = np.random.uniform(*self.rot_range)
        rotation = np.radians(rotation)

        trans_x = np.random.uniform(*self.trans_range)
        trans_y = np.random.uniform(*self.trans_range)
        trans_z = np.random.uniform(*self.trans_range)
        translation = (trans_x, trans_y, trans_z)

        return scale, rotation, translation

    def scale_transform(self, scale):
        transform = sitk.ScaleTransform(3)
        transform.SetScale(scale)

        return transform

    def ras_to_lps(self, triplet):
        return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)

    def euler_transform(self, rotation, translation):
        transform = sitk.Euler3DTransform()
        rad_lps = self.ras_to_lps(rotation)
        trans_lps = self.ras_to_lps(translation)
        transform.SetRotation(*rad_lps)
        transform.SetTranslation(trans_lps)

        return transform

    def composite_transform(self, transform_list):
        return sitk.CompositeTransform(transform_list)

    def apply(self, rt_img, gt):
        rt_sitk = sitk.GetImageFromArray(rt_img.astype(np.float32))
        gt_sitk = sitk.GetImageFromArray(gt.astype(np.float32))
        reference = rt_sitk

        scale, rotation, translation = self.get_params()

        scale_t = self.scale_transform(scale)
        euler_t = self.euler_transform(rotation, translation)
        transform_list = [scale_t, euler_t]
        transform = self.composite_transform(transform_list)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetDefaultPixelValue(float(rt_img.min()))
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)

        interpolator = get_interpolation(self.interpolation)
        resampler.SetInterpolator(interpolator)
        rt_resample = resampler.Execute(rt_sitk)
        rt_out = sitk.GetArrayFromImage(rt_resample)

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        gt_resample = resampler.Execute(gt_sitk)
        gt_out = sitk.GetArrayFromImage(gt_resample)

        rt_out[rt_out < 0] = 0
        gt_out[gt_out >= 0.5] = 1
        gt_out[gt_out < 0.5] = 0

        return (rt_out, gt_out)

class RandomAffineV2:
    def __init__(self, scale_range=(0.9,1,1), rot_range=(-90,90), trans_range=(-5,5), interpolation=1):
        self.scale_range = scale_range
        self.rot_range = rot_range
        self.trans_range = trans_range
        self.interpolation = interpolation

    def get_params(self):
        scale = np.random.uniform(*self.scale_range)
        rotation = np.random.uniform(*self.rot_range)
        translation = np.random.uniform(*self.trans_range)

        return scale, rotation, translation

    def affine_transform(self, img, scale, rotation, translation, interp):
        tf_img = img.copy()
        x, y, z = tf_img.shape
        try:
            tform = AffineTransform(scale=scale, translation=translation)
        except:
            tform = AffineTransform(scale=(scale,scale), translation=translation)
        for i in range(z):
            img_slice = tf_img[:,:,i].copy()
            img_slice = rotate(img_slice, rotation, order=interp, preserve_range=True)
            img_slice = warp(img_slice, tform.inverse, order=interp, preserve_range=True)
            tf_img[:,:,i] = img_slice.copy()

        return tf_img

    def apply(self, rt_img, gt):
        scale, rotation, translation = self.get_params()
        interp = self.interpolation

        rt_out = rt_img.copy()
        gt_out = gt.copy()

        rt_out = self.affine_transform(rt_out, scale, rotation, translation, interp)
        gt_out = self.affine_transform(gt_out, scale, rotation, translation, 0)

        return (rt_out, gt_out)

def rand_aug(rt_img: np.ndarray,
             gt: np.ndarray,
             transform_list: List[object]) -> Tuple[np.ndarray, np.ndarray]:
    """Random augmentation for brain MRI and its segmentation mask.

    Given image and mask, this will perfrom identical augmentation to both arrays and return them.

    Args:
        rt_img (np.ndarray): Image of brain MRI.
        gt (np.ndarray): Segmentation mask for brain MRI.
        transform_list (List['__class__']): List of transform classes.

    Returns:
        (np.ndarray, np.ndarray): Transformed brain MRI and its segmentation mask, respectively.

    """
    rt_out = rt_img.copy()
    gt_out = gt.copy()
    for transform in transform_list:
        rt_out, gt_out = transform.apply(rt_out, gt_out)

    return (rt_out, gt_out)