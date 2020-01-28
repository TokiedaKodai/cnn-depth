from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, concatenate, Lambda, add, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers

# lr = 0.01
# lr = 1e-6
# decay = 1e-6
# decay = 0

# Batch Normalization
is_batch_norm = True
# is_batch_norm = False

# DropOut
# is_drop_out = True
is_drop_out = False

# difference_scaling = 100

def build_network_model_difference_learn(batch_shape,
                                         ch_num,
                                         depth_threshold=0.1,
                                         difference_threshold=0.05,
                                        #  decay=0.0,
                                         drop_rate=0.12,
                                         scaling=100):
    def build_enc_block(input, ch):
        x = input

        x = Conv2D(ch, (3, 3), padding='same')(x)
        if is_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        x = Conv2D(ch*2, (3, 3), padding='same')(x)
        if is_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        if is_drop_out:
            x = Dropout(rate=drop_rate)(x)

        return x

    def build_dec_block(input, cnc, ch):
        x = input
        c = cnc

        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        if is_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        x = Conv2DTranspose(int(ch/2), (3, 3), padding='same')(x)
        if is_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        if is_drop_out:
            x = Dropout(rate=drop_rate)(x)

        return x
    
    # input_img = Input(shape=(*batch_shape, ch_num))
    input_img = Input(shape=(batch_shape[0], batch_shape[1], ch_num))

    c0 = build_enc_block(input_img, 16)
    p0 = MaxPooling2D((2, 2))(c0)

    c1 = build_enc_block(p0, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(256, (3, 3), padding='same')(p1)
    if is_batch_norm:
        c2 = BatchNormalization()(c2)
    c2 = Activation('tanh')(c2)
    ###
    d2 = Conv2DTranspose(128, (3, 3), padding='same')(c2)

    d1 = build_dec_block(d2, c1, 64)

    d0 = build_dec_block(d1, c0, 16)

    d0 = Conv2DTranspose(2, (3, 3), padding='same')(d0)
    if is_batch_norm:
        d0 = BatchNormalization()(d0)

    d0 = Dropout(rate=drop_rate)(d0) # new droopout
    
    output_img = Activation('tanh')(d0)

    def mean_squared_error_shading_only(y_true, y_pred):
        depth_gt = y_true[:, :, :, 0]
        depth_gap = y_true[:, :, :, 1]

        is_gt_available = depth_gt > depth_threshold
        is_gap_unavailable = depth_gap < depth_threshold

        is_depth_close = K.all(K.stack([
            K.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available
        ],
                                       axis=0),
                               axis=0)

        # difference learn
        gt = depth_gt - depth_gap

        # scale
        gt = gt * scaling

        # complement
        is_complement = False
        if is_complement:
            is_to_interpolate = K.all(K.stack(
                [is_gt_available, is_gap_unavailable], axis=0),
                                    axis=0)
            is_valid = K.any(K.stack([is_to_interpolate, is_depth_close], axis=0),
                            axis=0)
            # is_valid = K.cast(is_valid, float)
            is_valid = K.cast(is_valid, 'float32')
        else:
            # is_valid = K.cast(is_depth_close, float)
            is_valid = K.cast(is_depth_close, 'float32')

        valid_length = K.sum(is_valid)
        err = K.sum(K.square(gt - y_pred[:, :, :, 0]) * is_valid)
        return err / valid_length

    model = Model(input_img, output_img)
    # adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=mean_squared_error_shading_only)
    return model
