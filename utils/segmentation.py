import numpy as np
import tensorflow as tf

from typing import Optional
import tensorflow.keras.backend as K

def dwi_segmentation(rt_img: np.ndarray,
                     sgmt_model: tf.keras.Model,
                     clear_session: Optional[bool] = True) -> np.ndarray:
    """Lesion segmentation for diffusion-weighted image.

    Performs ischemic core segmentation from diffusion-weighted image.

    Args:
        rt_img (np.ndarray): 3D diffusion-weighted image stacked vertically
        sgmt_model (tf.keras.Model): Instance of tensorflow 2.0 model. Input shape must support rt_img.
        clear_session (Optional[bool]): This Parameter will determine whether to reload the model before prediction.

    Returns:
        np.ndarray: Model output of same shape as rt_img.

    """
    # Get no. of slices for model
    n_slice = sgmt_model.input_shape[3]

    # Slice centering
    model_in = np.zeros(sgmt_model.input_shape[1:])
    z_len = rt_img.shape[2]
    start = int((n_slice - z_len) / 2)
    model_in[:,:,start:start+z_len,0] = rt_img
    model_in = np.expand_dims(model_in, 0)

    # Segmentation
    if clear_session:
        K.clear_session()
        sgmt_model.predict(model_in, verbose=0)
    model_out = sgmt_model.predict(model_in, verbose=0)
    model_out = model_out[0,:,:,:,0]
    model_out = model_out[:,:,start:start+z_len]

    return model_out

def binarization(model_out: np.ndarray,
                 threshold: Optional[float] = 0.5) -> np.ndarray:
    """Model output binarization

    Binarizes model output by given threshold value.

    Args:
        model_out (np.ndarray): Model output.
        threshold(Optional[float]): Threshold value for binarizing model output.

    Returns:
        np.ndarray: Binarized model output.

    """
    lesion_mask = np.zeros(model_out.shape)
    lesion_mask[model_out >= threshold] = 1

    return lesion_mask