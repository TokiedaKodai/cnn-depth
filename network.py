from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
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

''' Dense-ResNet '''
def build_dense_resnet_model(batch_shape,
                            ch_num,
                            depth_threshold=0.1,
                            difference_threshold=0.05,
                            drop_rate=0.1,
                            scaling=100):
    ch = 16 # grow rate
    def encode_block(x):
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2D(ch, (1, 1), padding='same')(x)
        s = x
        x = base_block(x)
        x = base_block(x)
        x = add([x, s])
        return x
    
    def decode_block(x):
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x

        x = Conv2D(ch, (1, 1), padding='same')(x)
        s = x
        x = base_block(x)
        x = base_block(x)
        x = add([x, s])
        return x

    def conv_1x1(x, ch):
        x = Conv2D(ch, (1, 1), padding='same')(x)
        x = Activation('tanh')(x)
        return x

    input_batch = Input(shape=(*batch_shape, ch_num))

    r0 = conv_1x1(input_batch, 8)
    r0_1 = AveragePooling2D((2, 2))(r0)
    r0_2 = AveragePooling2D((2, 2))(r0_1)
    r0_3 = AveragePooling2D((2, 2))(r0_2)

    r1 = encode_block(r0)
    r1_1 = AveragePooling2D((2, 2))(r1)
    r1_2 = AveragePooling2D((2, 2))(r1_1)
    r1_3 = AveragePooling2D((2, 2))(r1_2)

    r2 = concatenate([r0_1, r1_1])
    r2 = encode_block(r2)
    r2_2 = AveragePooling2D((2, 2))(r2)
    r2_3 = AveragePooling2D((2, 2))(r2_2)
    r2_0 = UpSampling2D((2, 2))(r2)

    r3 = concatenate([r0_2, r1_2, r2_2])
    r3 = encode_block(r3)
    r3_3 = AveragePooling2D((2, 2))(r3)
    r3_1 = UpSampling2D((2, 2))(r3)
    r3_0 = UpSampling2D((2, 2))(r3_1)

    r4 = concatenate([r0_3, r1_3, r2_3, r3_3])
    r4 = encode_block(r4)
    r4_2 = UpSampling2D((2, 2))(r4)
    r4_1 = UpSampling2D((2, 2))(r4_2)
    r4_0 = UpSampling2D((2, 2))(r4_1)

    r5 = concatenate([r0_2, r1_2, r2_2, r3, r4_2])
    r5 = decode_block(r5)
    r5_1 = UpSampling2D((2, 2))(r5)
    r5_0 = UpSampling2D((2, 2))(r5_1)

    r6 = concatenate([r0_1, r1_1, r2, r3_1, r4_1, r5_1])
    r6 = decode_block(r6)
    r6_0 = UpSampling2D((2, 2))(r6)

    r7 = concatenate([r0, r1, r2_0, r3_0, r4_0, r5_0, r6_0])
    r7 = decode_block(r7)

    r8 = conv_1x1(r7, 8)

    output_batch = conv_1x1(r8, 2)

    def mean_squared_error_difference_learn(y_true, y_pred):
        depth_gt = y_true[:, :, :, 0]
        depth_gap = y_true[:, :, :, 1]

        is_gt_available = depth_gt > depth_threshold
        is_gap_unavailable = depth_gap < depth_threshold

        is_depth_close = K.all(K.stack([
            K.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available], axis=0), axis=0)

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

    model = Model(input_batch, output_batch)
    # adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=mean_squared_error_difference_learn)
    return model

''' U-ResNet '''
def build_resnet_model(batch_shape,
                        ch_num,
                        depth_threshold=0.1,
                        difference_threshold=0.05,
                        drop_rate=0.1,
                        scaling=100):
    def encode_block(x, ch):
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            return x
        
        s = Conv2D(ch, (1, 1), padding='same')(x)
        x = base_block(x)
        x = base_block(x)
        x = add([x, s])
        return x
    
    def decode_block(x, c, ch):
        ch = ch
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        s = Conv2D(ch, (1, 1), padding='same')(x)
        x = base_block(x)
        x = base_block(x)
        x = add([x, s])
        return x
    
    input_batch = Input(shape=(*batch_shape, ch_num))

    e0 = AveragePooling2D((2, 2))(input_batch)
    e0 = UpSampling2D((2, 2))(e0)
    e0 = Conv2D(8, (1, 1), padding='same')(e0)

    # e0 = Conv2D(8, (1, 1), padding='same')(input_batch)
    e0 = Activation('tanh')(e0)

    e0 = encode_block(e0, 16)

    e1 = AveragePooling2D((2, 2))(e0)
    e1 = encode_block(e1, 32)

    e2 = AveragePooling2D((2, 2))(e1)
    e2 = encode_block(e2, 64)

    e3 = AveragePooling2D((2, 2))(e2)
    e3 = encode_block(e3, 128)

    d2 = decode_block(e3, e2, 64)
    d1 = decode_block(d2, e1, 32)
    d0 = decode_block(d1, e0, 16)

    d0 = Conv2D(2, (1, 1), padding='same')(d0)
    output_batch = Activation('tanh')(d0)

    def mean_squared_error_difference_learn(y_true, y_pred):
        depth_gt = y_true[:, :, :, 0]
        depth_gap = y_true[:, :, :, 1]

        is_gt_available = depth_gt > depth_threshold
        is_gap_unavailable = depth_gap < depth_threshold

        is_depth_close = K.all(K.stack([
            K.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available], axis=0), axis=0)

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

    model = Model(input_batch, output_batch)
    # adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=mean_squared_error_difference_learn)
    return model


''' U-Net '''
def build_unet_model(batch_shape,
                        ch_num,
                        depth_threshold=0.1,
                        difference_threshold=0.05,
                        drop_rate=0.1,
                        scaling=100,
                        transfer_learn=False):
    def encode_block(x, ch):
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2D(ch, (3, 3), padding='same')(x)
            return x
        
        x = base_block(x)
        x = base_block(x)
        return x
    
    def decode_block(x, c, ch):
        ch = ch
        def base_block(x):
            x = BatchNormalization()(x)
            x = Activation('tanh')(x)
            x = Dropout(rate=drop_rate)(x)
            x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
            return x
        
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        x = base_block(x)
        x = base_block(x)
        return x
    
    input_batch = Input(shape=(*batch_shape, ch_num))
    e0 = Conv2D(8, (1, 1), padding='same')(input_batch)
    e0 = Activation('tanh')(e0)

    e0 = encode_block(e0, 16)

    e1 = AveragePooling2D((2, 2))(e0)
    e1 = encode_block(e1, 32)

    e2 = AveragePooling2D((2, 2))(e1)
    e2 = encode_block(e2, 64)

    e3 = AveragePooling2D((2, 2))(e2)
    e3 = encode_block(e3, 128)

    d2 = decode_block(e3, e2, 64)
    d1 = decode_block(d2, e1, 32)
    d0 = decode_block(d1, e0, 16)

    d0 = Conv2D(2, (1, 1), padding='same')(d0)
    output_batch = Activation('tanh')(d0)

    def mean_squared_error_difference_learn(y_true, y_pred):
        depth_gt = y_true[:, :, :, 0]
        depth_gap = y_true[:, :, :, 1]

        is_gt_available = depth_gt > depth_threshold
        is_gap_unavailable = depth_gap < depth_threshold

        is_depth_close = K.all(K.stack([
            K.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available], axis=0), axis=0)

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

    model = Model(input_batch, output_batch)

    # Transfer Learning
    if transfer_learn:
        for l in model.layers[:38]:
            l.trainable = False

    # adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss=mean_squared_error_difference_learn)
    return model


''' U-Net old '''
def build_unet_model_old(batch_shape,
                    ch_num,
                    depth_threshold=0.1,
                    difference_threshold=0.05,
                #  decay=0.0,
                    drop_rate=0.12,
                    scaling=100):
    def build_enc_block(input, ch):
        x = input

        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(rate=drop_rate)(x)
        x = Conv2D(ch, (3, 3), padding='same')(x)

        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(rate=drop_rate)(x)
        x = Conv2D(ch*2, (3, 3), padding='same')(x)

        return x

    def build_dec_block(input, cnc, ch):
        x = input
        c = cnc

        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, c])

        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(rate=drop_rate)(x)
        x = Conv2DTranspose(ch, (3, 3), padding='same')(x)

        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(rate=drop_rate)(x)
        x = Conv2DTranspose(int(ch/2), (3, 3), padding='same')(x)

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

    def mean_squared_error_difference_learn(y_true, y_pred):
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
                  loss=mean_squared_error_difference_learn)
    return model


''' U-Net old '''
def build_unet_model_old_1(batch_shape,
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

    def mean_squared_error_difference_learn(y_true, y_pred):
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
                  loss=mean_squared_error_difference_learn)
    return model
