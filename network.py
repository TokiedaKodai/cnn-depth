from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, concatenate, Lambda, add, Dropout
from keras.models import Model
import keras.backend as K
from keras import optimizers

# lr = 0.00001
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

depth_threshold = 0.2
difference_threshold = 0.01
difference_threshold = 10

''' U-Net '''
def build_unet_model(batch_shape,
                    ch_num,
                    drop_rate=0.1,
                    transfer_learn=False,
                    transfer_encoder=False,
                    lr=0.001,
                    scaling=1
                    ):
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

    # d0 = Conv2D(2, (1, 1), padding='same')(d0)
    # output_batch = Activation('tanh')(d0)
    # output_batch = Conv2D(2, (1, 1), padding='same')(d0)
    output_batch = Conv2D(1, (1, 1), padding='same')(d0)

    def mean_squared_error_with_mask(y_true, y_pred):
        difference = y_true[:, :, :, 0]
        depth_gap = y_true[:, :, :, 1]
        # mask = y_true[:, :, :, 1]

        difference *= scaling

        is_gap_available = depth_gap > depth_threshold
        # is_depth_close = difference < difference_threshold
        is_depth_close = K.all(K.stack([K.abs(difference) < difference_threshold, 
                                        is_gap_available], axis=0), axis=0)
        mask = K.cast(is_depth_close, 'float32')

        mask_length = K.sum(mask)
        err = K.sum(K.square(difference - y_pred[:, :, :, 0]) * mask) / mask_length # MSE
        # err = K.mean(K.square(difference - y_pred[:, :, :, 0]) * mask)
        # err = K.sum(K.abs(difference - y_pred[:, :, :, 0]) * mask) / mask_length # MAE
        return err

    model = Model(input_batch, output_batch)

    # Transfer Learning
    if transfer_learn:
        for l in model.layers[:38]:
            l.trainable = False
    elif transfer_encoder:
        for l in model.layers[38:]:
            l.trainable = False

    # adam = optimizers.Adam(lr=lr, decay=decay)
    adam = optimizers.Adam(lr=lr)
    model.compile(
                # optimizer='adam',
                optimizer=adam,
                metrics=['accuracy'],
                loss=mean_squared_error_with_mask
                # loss='mean_squared_error'
                # loss='mean_absolute_error'
                )
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
    # e3 = encode_block(e2, 128)
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
        # err = K.sum(K.square(gt - y_pred[:, :, :, 0]) * is_valid)  / valid_length # MSE
        err = K.sum(K.abs(gt - y_pred[:, :, :, 0]) * is_valid)  / valid_length # MAE
        return err

    model = Model(input_batch, output_batch)
    # adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                #   loss=mean_squared_error_difference_learn
                #   loss='mean_squared_error'
                  loss='mean_absolute_error'
                  )
    return model
