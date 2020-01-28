import depth_tools
import cv2
import numpy as np
import pandas as pd
from itertools import product
import os
from tqdm import tqdm
from glob import glob
import re
import sys
import os
import random

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint

import shade_cnn_common as scc

'''
ARGV
1: local output dir
2: is model exists (is not training start)
3: epoch num
4-6: parameter
'''
argv = sys.argv
_, out_local, is_model_exist, epoch_num, augment_rate, augment_type, dropout = argv

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set currenct dir

#InputData
src_dir = '../data/input_200117'

#RemoteOutput
out_dir = 'output'
#LocalOutput
# Train on local
is_local_train = True
# is_local_train = False
if is_local_train:
    out_dir = '../output/' + out_local

# resume_from = None  # start new without resume
# resume_from = 'auto'  # resume from latest model file
# resume_from = ['output /model-final.hdf5', 5]   # [file, epoch_num]

'''
Test Data
100cm : 32 - 47
'''
'''data index'''
# data_idx_range = list(range(32))
# data_idx_range.extend(list(range(48, 80))) 

'''no-fake data'''
# data_idx_range = list(range(12))
# data_idx_range.extend(list(range(16, 28)))
# data_idx_range.extend(list(range(32, 44)))
# data_idx_range.extend(list(range(48, 60)))
# data_idx_range.extend(list(range(64, 76)))

'''no-rotate data'''
# data_idx_range = list()
# for i in range(5):
#     data_idx_range.extend(list(range(0 + 16*i, 6 + 16*i)))
#     data_idx_range.extend(list(range(12 + 16*i, 14 + 16*i)))

'''data distance 80,90,100 cm'''
data_idx_range = range(48)

# parameters
depth_threshold = 0.2
difference_threshold = 0.01
patch_remove = 0.5
# dropout_rate = 0.12
dropout_rate = int(dropout) / 100

# model
save_period = 10

# input
is_input_depth = True
is_input_frame = True

# normalization
is_shading_norm = True

batch_shape = (120, 120)
batch_tl = (0, 0)  # top, left

train_batch_size = 64
# val_rate = 0.1
val_rate = 0.3

# progress bar
verbose = 1

# augmentation
is_augment = True
# is_augment = False

# zoom_range=[0.8, 1.2]
zoom_range=[0.8, 1.0]

augment_rate = int(augment_rate)
if augment_rate == 0:
    is_augment = False

def augment_zoom(img):
    h, w, s = img.shape
    random.seed(int(np.sum(img[:, :, 1])))
    scale = random.uniform(zoom_range[0], zoom_range[1])
    resize_w, resize_h = int(w*scale), int(h*scale)
    
    x = cv2.resize(img, (resize_w, resize_h))
    x[:, :, 1] = x[:, :, 1] / scale
    if s is 2:
        x[:, :, 0] = x[:, :, 0] / scale

    if scale > 1:
        new_img = x[int((resize_h - h)//2): int((resize_h + h)//2),
                   int((resize_w - w)//2): int((resize_w + w)//2), :]
    else:
        new_img = np.zeros_like(img)
        new_img[int((h - resize_h)//2): int((h + resize_h)//2),
                int((w - resize_w)//2): int((w + resize_w)//2), :] = x
    return new_img

if is_augment:
    if augment_type is '0':
        datagen_args = dict(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0,
            # zoom_range=[0.8, 1.2],
            # zoom_range=[0.9, 1],
            # zoom_range=[0.9, 1.1],
            fill_mode='constant',
            cval=0,
            # horizontal_flip=True,
            # vertical_flip=True
            preprocessing_function=augment_zoom
        )
    elif augment_type is '1': # shift
        datagen_args = dict(
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )
    elif augment_type is '2': # rotate
        datagen_args = dict(
            rotation_range=45,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )
    elif augment_type is '3': # zoom
        datagen_args = dict(
            shear_range=0,
            fill_mode='constant',
            cval=0,
            preprocessing_function=augment_zoom
        )
    elif augment_type is '4': # no-zoom
        datagen_args = dict(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )

    x_datagen = ImageDataGenerator(**datagen_args)
    y_datagen = ImageDataGenerator(**datagen_args)
    x_val_datagen = ImageDataGenerator(**datagen_args)
    y_val_datagen = ImageDataGenerator(**datagen_args)
    seed_train = 1
    seed_val = 2


epoch_num = int(epoch_num)


if is_model_exist is '0':
    resume_from = None  # start new without resume
else:
    # resume_from = 'auto'  # resume from latest model file

    df_log = pd.read_csv(out_dir + '/training.log')
    pre_epoch = int(df_log.tail(1).index.values) + 1
    resume_from = pre_epoch 


def zip(*iterables):
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)


def prepare_data(data_idx_range):
    def clip_batch(img, top_left, size):
        # t, l, h, w = *top_left, *size
        t = top_left[0]
        l = top_left[1]
        h = size[0]
        w = size[1]
        return img[t:t + h, l:l + w]

    src_rec_dir = src_dir + '/rec'
    src_frame_dir = src_dir + '/frame'
    src_gt_dir = src_dir + '/gt'
    src_shading_dir = src_dir + '/shading'

    # read data
    print('loading data...')
    x_train = []
    y_train = []
    for data_idx in tqdm(data_idx_range):
        src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
        src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(data_idx)
        src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
        src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        shading = cv2.imread(src_shading, -1)
        if is_shading_norm:
            shading = shading / np.max(shading)
        else:
            shading = shading / 255.


        # normalization (may not be needed)
        # depth_gap /= depth_gap.max()
        # depth_gt /= depth_gt.max()

        depth_thre = depth_threshold

        # merge bgr + depth_gap
        if is_input_frame:
            bgrd = np.dstack([shading[:, :, 0], depth_gap, bgr[:, :, 0]])
        else:
            bgrd = np.dstack([shading[:, :, 0], depth_gap])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(batch_tl[0], img_shape[0], batch_shape[0])
        left_coords = range(batch_tl[1], img_shape[1], batch_shape[1])

        # add training data
        for top, left in product(top_coords, left_coords):
            batch_train = clip_batch(bgrd, (top, left), batch_shape)

            # do not add batch if not valid ################
            # valid_pixels = np.logical_and(
            #     batch_train[:, :, 0].mean() > 0,
            #     batch_train[:, :, 1] > depth_threshold)
            # if np.count_nonzero(valid_pixels) < (b_h * b_w * 0.5):
            #     continue

            batch_gt_depth = clip_batch(depth_gt, (top, left), batch_shape)
            batch_gt_mask = clip_batch(depth_gap, (top, left), batch_shape)
            batch_gt = np.dstack([batch_gt_depth, batch_gt_mask])

            # do not add batch if not close ################
            is_gt_available = batch_gt_depth > depth_thre
            is_depth_close = np.logical_and(
                np.abs(batch_train[:, :, 1] - batch_gt_depth) < difference_threshold,
                is_gt_available)
            if np.count_nonzero(is_depth_close) < (b_h * b_w * patch_remove):
                continue

            if is_input_depth or is_input_frame:
                x_train.append(batch_train)
            else:
                # x_train.append(batch_train[:, :, 0].reshape((*batch_shape, 1)))
                x_train.append(batch_train[:, :, 0].reshape((batch_shape[0], batch_shape[1], 1)))
            # y_train.append(batch_gt.reshape((*batch_shape, 2)))
            y_train.append(batch_gt.reshape((batch_shape[0], batch_shape[1], 2)))
    return np.array(x_train), np.array(y_train), depth_thre


def main():
    x_data, y_data, depth_thre = prepare_data(data_idx_range)
    print('x train data:', x_data.shape)
    print('y train data:', y_data.shape)

    if is_augment:
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=val_rate)
    else:
        x_train, y_train = x_data, y_data

    # check training data
    print('x_train length:', len(x_train))

    if is_augment:
        print('x_val length  :', len(x_val))
        print('data augmentation')
        x_datagen.fit(x_train, augment=True, seed=seed_train)
        y_datagen.fit(y_train, augment=True, seed=seed_train)
        x_val_datagen.fit(x_val, augment=True, seed=seed_val)
        y_val_datagen.fit(y_val, augment=True, seed=seed_val)

        x_generator = x_datagen.flow(x_train, batch_size=train_batch_size, seed=seed_train)
        y_generator = y_datagen.flow(y_train, batch_size=train_batch_size, seed=seed_train)
        x_val_generator = x_val_datagen.flow(x_val, batch_size=train_batch_size, seed=seed_val)
        y_val_generator = y_val_datagen.flow(y_val, batch_size=train_batch_size, seed=seed_val)

        train_generator = zip(x_generator, y_generator)
        val_generator = zip(x_val_generator, y_val_generator)
        print('generator created')

    if is_input_depth:
        if is_input_frame:
            ch_num = 3
        else:
            ch_num = 2
    else:
        if is_input_frame:
            ch_num = 2
        else:
            ch_num = 1

    # model configuration
    model = scc.build_network_model_difference_learn(
        batch_shape,
        ch_num,
        depth_threshold=depth_thre,
        difference_threshold=difference_threshold,
        # decay=decay,
        drop_rate=dropout_rate,
        # scaling=difference_scaling
        )
    

    # resume
    model_dir = out_dir + '/model'
    initial_epoch = 0
    if resume_from is None:
        initial_epoch = 0

    elif resume_from == 'auto':

        def extract_epoch_num(filename):
            return int(re.search(r'model-([0-9]+)\.hdf5', filename).group(1))

        model_files = glob(model_dir + '/model-*.hdf5')
        model_file = sorted(model_files, key=extract_epoch_num)[-1]
        initial_epoch = extract_epoch_num(model_file)
        print('resume from ', model_file, ', epoch number', initial_epoch)

    else:
        # model_file, initial_epoch = resume_from
        initial_epoch = resume_from
        # model_file = model_dir + '/model-best.hdf5'
        model_file = model_dir + '/model-final.hdf5'
        # print('resume from ', model_file, ', epoch number', initial_epoch)

    if resume_from is not None:
        model.load_weights(model_file)

    # make output dirs
    if is_model_exist is '0':
        os.makedirs(out_dir)
        os.makedirs(model_dir)

    # train
    model_save_cb = ModelCheckpoint(model_dir + '/model-best.hdf5',
                                    # model_dir + '/model-{epoch:03d}.hdf5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='min',
                                    period=1)
    # model_save_cb = ModelCheckpoint(model_dir + '/model-{epoch:03d}.hdf5',
    #                                 period=save_period,
    #                                 save_weights_only=True)
    csv_logger_cb = CSVLogger(out_dir + '/training.log',
                              append=(resume_from is not None))

    print('training')
    if is_augment:
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(x_train)*augment_rate / train_batch_size + 1,
            epochs=epoch_num,
            initial_epoch=initial_epoch,
            shuffle=True,
            callbacks=[model_save_cb, csv_logger_cb],
            validation_data=val_generator,
            validation_steps=len(x_val)*augment_rate / train_batch_size + 1,
            verbose=verbose)
    else:
        model.fit(
            x_train,
            y_train,
            epochs=epoch_num,
            batch_size=64,
            initial_epoch=initial_epoch,
            shuffle=True,
            validation_split=val_rate,
            callbacks=[model_save_cb, csv_logger_cb],
            verbose=verbose)

    model.save_weights(model_dir + '/model-final.hdf5')


if __name__ == "__main__":
    main()
