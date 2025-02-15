import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

def combined_dice_loss(y_true, y_pred):
    # One-hot encoding
    y_true_oh = tf.one_hot(tf.cast(y_true[..., 0], tf.int32), depth=4)
    
    num = 2. * tf.reduce_sum(y_true_oh * y_pred, axis=(1, 2))
    den = tf.reduce_sum(y_true_oh + y_pred, axis=(1, 2))
    
    dice_per_class = (num + 1e-6) / (den + 1e-6)
    
    # force model to care about the tumuors and not all with bg
    relevant_dice = dice_per_class[:, 1:] 
    
    return 1 - tf.reduce_mean(relevant_dice)

def build_unet(input_shape=(128, 128, 4), num_classes=4):
    def conv_block(x, filters):
        for _ in range(2):
            x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        return x

    inputs = Input(input_shape)
    
    c1 = conv_block(inputs, 16); p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 32); p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 64); p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 128); p4 = MaxPooling2D((2, 2))(c4)
   
    c5 = conv_block(p4, 256)
 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4]); c6 = conv_block(u6, 128)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3]); c7 = conv_block(u7, 64)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2]); c8 = conv_block(u8, 32)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1]); c9 = conv_block(u9, 16)

    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(c9)
    return Model(inputs, outputs)