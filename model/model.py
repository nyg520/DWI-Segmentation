from tensorflow.keras.layers import (
    MaxPooling3D,
    Conv3DTranspose,
    Concatenate,
    Conv3D
    )
from tensorflow.keras.regularizers import l2

from .layer import bn_relu_conv

def UNet_encoder(x, kernel_size=3, nb_filter=[32,64,128,256,512], reg_val=1e-4, act='relu'):

    # Downscaling
    conv1 = bn_relu_conv(x, nb_filter[0], kernel_size, reg_val=reg_val, act=act)
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)

    conv2 = bn_relu_conv(pool1, nb_filter[1], kernel_size, reg_val=reg_val, act=act)
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)

    conv3 = bn_relu_conv(pool2, nb_filter[2], kernel_size, reg_val=reg_val, act=act)
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)

    conv4 = bn_relu_conv(pool3, nb_filter[3], kernel_size, reg_val=reg_val, act=act)
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv4)

    conv5 = bn_relu_conv(pool4, nb_filter[4], kernel_size, reg_val=reg_val, act=act)

    return [conv1, conv2, conv3, conv4, conv5]

def UNet_decoder(x, kernel_size=3, nb_filter=[32,64,128,256,512], reg_val=1e-4, act='relu', channel=1):

    # Unpacking
    conv1, conv2, conv3, conv4, conv5 = x

    # Upscaling
    up4 = Conv3DTranspose(nb_filter[3], kernel_size=2, strides=(2, 2, 2),
                          padding='same')(conv5)
    concat4 = Concatenate()([up4, conv4])
    upconv4 = bn_relu_conv(concat4, nb_filter[3], kernel_size, reg_val=reg_val, act=act)

    up3 = Conv3DTranspose(nb_filter[2], kernel_size=2, strides=(2, 2, 2),
                          padding='same')(upconv4)
    concat3 = Concatenate()([up3, conv3])
    upconv3 = bn_relu_conv(concat3, nb_filter[2], kernel_size, reg_val=reg_val, act=act)

    up2 = Conv3DTranspose(nb_filter[1], kernel_size=2, strides=(2, 2, 2),
                          padding='same')(upconv3)
    concat2 = Concatenate()([up2, conv2])
    upconv2 = bn_relu_conv(concat2, nb_filter[1], kernel_size, reg_val=reg_val, act=act)

    up1 = Conv3DTranspose(nb_filter[0], kernel_size=2, strides=(2, 2, 2),
                          padding='same')(upconv2)
    concat1 = Concatenate()([up1, conv1])
    upconv1 = bn_relu_conv(concat1, nb_filter[0], kernel_size, reg_val=reg_val, act=act)

    output = Conv3D(channel, 1, activation='sigmoid', kernel_initializer='he_normal',
                    padding='same', kernel_regularizer=l2(reg_val))(upconv1)

    return output