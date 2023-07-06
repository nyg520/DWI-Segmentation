from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

def bn_relu_conv(x, nb_filter, kernel_size, reg_val, act='relu'):

    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Conv3D(nb_filter, kernel_size, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=l2(reg_val))(x)

    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Conv3D(nb_filter, kernel_size, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=l2(reg_val))(x)

    return x