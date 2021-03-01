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
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import network

'''
ARGV
1: local output dir
2: is model exists (is not training start)
3: epoch num
4-6: parameter
'''
argv = sys.argv
# _, out_local, is_model_exist, epoch_num, augment_type, dropout, net_type, transfer_learn = argv
# _, out_local, learn_type, start, epoch_num, augment_type, dropout, learning_rate, norm = argv
# _, out_local, learn_type, start, epoch_num, augment_type, dropout, scale = argv
# _, out_local, learn_type, start, epoch_num, augment_type, dropout = argv
# out_local, learn_type, start, epoch_num, augment_type, dropout = 'wave1', '0', '0', '300', '0', '10'
_, out_local, learn_type, start, epoch_num, augment_type, dropout, max_rotate = argv

# learning_rate = '1'
# if learning_rate is '0':
#     learning_rate = 0.001
# elif learning_rate is '1':
#     learning_rate = 0.01
# elif learning_rate is '2':
#     learning_rate = 0.00001

learning_rate = 0.001 # Default
# learning_rate = 0.0001

# Transfer Learning
is_transfer_learning = False
is_finetune = False
is_transfer_encoder = False
if learn_type is '1':
    is_transfer_learning = True
elif learn_type is '2':
    is_finetune = True
elif learn_type is '3':
    is_transfer_encoder = True


is_start = True
if start is '1':
    is_start = False

# if is_transfer_learning or is_finetune:
if is_transfer_learning or is_transfer_encoder:
    is_model_exist = '1'
else:
    if is_start:
        is_model_exist = '0'
    else:
        is_model_exist = '1'

net_type = '0'

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set currenct dir

is_from_min = False
is_select_val = True

is_use_from_real = False

#InputData
if is_transfer_learning or is_finetune or is_transfer_encoder:
    src_dir = '../data/input_200318'
    src_dir = '../data/board-real'
    # src_dir = '../data/real'
    use_generator = False
    # is_from_min = True
else:
    use_generator = False
    # use_generator = True
    src_dir = '../data/render'
    src_dir = '../data/render_wave1_300'
    # src_dir = '../data/render_wave1-board'
    src_dir = '../data/render_wave1-double'
    src_dir = '../data/render_wave1-double_800'
    # src_dir = '../data/render_wave1-direct'
    # src_dir = '../data/render_wave1-double-direct'
    src_dir = '../data/render_wave2_1100'
    # src_dir = '../data/render_wave2-board'
    # src_dir = '../data/render_wave2-direct'
    src_dir = '../data/render_waves_600'

    src_dir = '../data/render_wave1-pose_200'
    src_dir = '../data/render_wave1d-pose_400'
    src_dir = '../data/render_wave2-pose_600'

    is_use_from_real = False
    src_2_dir = '../data/render_from-real'

#RemoteOutput
out_dir = 'output'
#LocalOutput
# Train on local
is_local_train = True
# is_local_train = False
out_local = 'output_' + out_local
if is_local_train:
    out_dir = '../output/' + out_local

# resume_from = None  # start new without resume
# resume_from = 'auto'  # resume from latest model file
# resume_from = ['output /model-final.hdf5', 5]   # [file, epoch_num]

'''
Test Data
ori 110cm : 16 - 23
small 100cm : 44 - 47
mid 110cm : 56 - 59
'''
'''data index'''
# data_idx_range = range(80)
# data_idx_range = range(160)

if is_transfer_learning or is_finetune or is_transfer_encoder:
    # data_idx_range = list(range(16)) # 0 - 15
    # data_idx_range.extend(list(range(24, 40))) # 24 - 43
    # data_idx_range.extend(list(range(40, 44)))
    # data_idx_range.extend(list(range(48, 56))) # 48 - 55
    # data_idx_range.extend(list(range(60, 68))) # 60 -67

    # data_idx_range = [0, 1, 3, 6, 40, 41, 42, 43, 48, 49, 50, 51]
    # data_idx_range = [0, 1, 3, 6, 16, 17, 19, 22]
    # data_idx_range = range(40)
    data_idx_range = range(16)
    data_idx_range = range(21)
    data_idx_range = range(25)
    data_idx_range = range(27)
else:
    data_idx_range = range(160)
    data_idx_range = range(200)
    data_idx_range = range(400)
    data_idx_range = range(600)
    # data_idx_range = range(800)

# parameters
depth_threshold = 0.2
difference_threshold = 0.01
# difference_threshold = 0.005
# difference_threshold = 0.003
patch_remove = 0.5
patch_remove = 0.9
# dropout_rate = 0.12
dropout_rate = int(dropout) / 100

# model
save_period = 1

# monitor train loss or val loss
monitor_loss = 'val_loss'
# monitor_loss = 'loss'

# input
is_input_depth = True
is_input_frame = True
# is_input_frame = False
is_input_coord = False
# is_input_coord = True

# normalization
is_shading_norm = True # Shading Normalization
is_shading_norm = False
is_difference_norm = True # Difference Normalization
# is_difference_norm = False

# Shading Noise
is_noise_shading = True
is_noise_shading = False
freq_num = 1
freq_range = 100
noise_intensity = 50
# GT Noise
is_noise_gt = True
# is_noise_gt = False
noise_mean = 0
noise_sigma = 0.0005


batch_shape = (120, 120)
batch_tl = (0, 0)  # top, left
img_size = 1200

train_batch_size = 64

# val_rate = 0.1
val_rate = 0.3

# progress bar
verbose = 1
# verbose = 2

# train_std = 0.0019195375434992092

train_std = 0.0014782568217017 # 1 wave



# difference_scaling = 100
# difference_scaling = 1000
difference_scaling = 1
# difference_scaling = int(scale)
# difference_scaling = 1 / train_std # Scale by SD

# augmentation
is_augment = True
if augment_type == '0':
    is_augment = False
augment_rate = 1
# augment_rate = 4
augment_val_rate = 1
# augment_val_rate = 4

shift_max = 0.1
shift_max = 0.2
# shift_max = 0.5
rotate_max = int(max_rotate)
# rotate_max = 45
# rotate_max = 90
# zoom_range=[0.5, 1.5]
zoom_range=[0.9, 1.1]
zoom_range=[0.8, 1.2]

lumi_scale_range = [0.5, 1.5]
# lumi_scale_range = [0.2, 5]

if is_transfer_learning or is_finetune or is_transfer_encoder:
    difference_threshold = 0.005


def perlin(r,seed=np.random.randint(0,100)):
    def fade(t):return 6*t**5-15*t**4+10*t**3
    def lerp(a,b,t):return a+fade(t)*(b-a)

    np.random.seed(seed)

    ri = np.floor(r).astype(int)
    ri[0] -= ri[0].min()
    ri[1] -= ri[1].min()
    rf = np.array(r) % 1
    g = 2 * np.random.rand(ri[0].max()+2,ri[1].max()+2,2) - 1
    e = np.array([[[[0,0],[0,1],[1,0],[1,1]]]])
    er = (np.array([rf]).transpose(2,3,0,1) - e).reshape(r.shape[1],r.shape[2],4,1,2)
    gr = np.r_["3,4,0",g[ri[0],ri[1]],g[ri[0],ri[1]+1],g[ri[0]+1,ri[1]],g[ri[0]+1,ri[1]+1]].transpose(0,1,3,2).reshape(r.shape[1],r.shape[2],4,2,1)
    p = (er@gr).reshape(r.shape[1],r.shape[2],4).transpose(2,0,1)

    return lerp(lerp(p[0],p[2],rf[0]),lerp(p[1],p[3],rf[0]),rf[1])

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

def augment_luminance(img):
    aug_scale = random.uniform(lumi_scale_range[0], lumi_scale_range[1])
    img[:, :, 0] *= aug_scale
    if is_input_frame:
        img[:, :, 2] *= aug_scale
    return img

if is_augment:
    if augment_type is '1':
        datagen_args = dict(
            rotation_range=rotate_max,
            width_shift_range=shift_max,
            height_shift_range=shift_max,
            shear_range=0,
            # zoom_range=[0.9, 1.1],
            fill_mode='constant',
            cval=0,
            # horizontal_flip=True,
            # vertical_flip=True
            preprocessing_function=augment_zoom
        )
    elif augment_type is '2': # shift
        datagen_args = dict(
            width_shift_range=shift_max,
            height_shift_range=shift_max,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )
    elif augment_type is '3': # rotate
        datagen_args = dict(
            rotation_range=rotate_max,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )
    elif augment_type is '4': # zoom
        datagen_args = dict(
            shear_range=0,
            fill_mode='constant',
            cval=0,
            preprocessing_function=augment_zoom
        )
    elif augment_type is '5': # no-zoom
        datagen_args = dict(
            rotation_range=rotate_max,
            width_shift_range=shift_max,
            height_shift_range=shift_max,
            shear_range=0,
            fill_mode='constant',
            cval=0,
        )
    elif augment_type is '6': # no-rotate
        datagen_args = dict(
            width_shift_range=shift_max,
            height_shift_range=shift_max,
            shear_range=0,
            fill_mode='constant',
            cval=0,
            preprocessing_function=augment_zoom
        )
    elif augment_type is '7': # no-scale
        datagen_args = dict(
            rotation_range=rotate_max,
            width_shift_range=shift_max,
            height_shift_range=shift_max,
            shear_range=0,
            zoom_range=zoom_range,
            fill_mode='constant',
            cval=0
        )
    elif augment_type is '8': # shading, pattern luminance aug
        datagen_args = dict(
            rotation_range=rotate_max,
            preprocessing_function=augment_luminance
        )

    x_datagen = ImageDataGenerator(**datagen_args)
    # y_datagen = ImageDataGenerator(**datagen_args)
    # x_datagen = ImageDataGenerator() # train loss no-aug
    y_datagen = ImageDataGenerator() # train loss no-aug
    x_val_datagen = ImageDataGenerator(**datagen_args) # val aug
    # y_val_datagen = ImageDataGenerator(**datagen_args) # val aug
    # x_val_datagen = ImageDataGenerator()
    y_val_datagen = ImageDataGenerator()
    seed_train = 1
    seed_val = 2


epoch_num = int(epoch_num)


if is_model_exist is '0':
    resume_from = None  # start new without resume
else:
    # resume_from = 'auto'  # resume from latest model file

    if is_start:
        resume_from = 0
    else:
        df_log = pd.read_csv(out_dir + '/training.log')
        pre_epoch = int(df_log.tail(1).index.values) + 1

        if is_from_min:
            resume_from = 'from_min'
            if is_select_val:
                df = df_log['val_loss']
            else:
                df = df_log['loss']
            df.index = df.index + 1
            idx_min_loss = df.idxmin()
        else:
            # resume_from = pre_epoch
            resume_from = 'auto'


# def zip(*iterables):
#     sentinel = object()
#     iterators = [iter(it) for it in iterables]
#     while iterators:
#         result = []
#         for it in iterators:
#             elem = next(it, sentinel)
#             if elem is sentinel:
#                 return
#             result.append(elem)
#         yield tuple(result)


def prepare_data(data_idx_range, return_valid=False):
    def clip_batch(img, top_left, size):
        t, l, h, w = *top_left, *size
        # t = top_left[0]
        # l = top_left[1]
        # h = size[0]
        # w = size[1]
        return img[t:t + h, l:l + w]

    

    if is_transfer_learning or is_finetune or is_transfer_encoder:
        src_rec_dir = src_dir + '/rec'
        # src_rec_dir = src_dir + '/rec_ajusted'
        # src_rec_dir = src_dir + '/lowres' # Median Filter Depth
        src_frame_dir = src_dir + '/frame'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shading'
    else:
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'

    # read data
    # print('loading data...')
    data_idx_range = list(data_idx_range)
    x_train = []
    y_train = []
    valid = []
    for data_idx in tqdm(data_idx_range):
    # for data_idx in data_idx_range:
        if return_valid:
            print('{:04d} : {:04d} - {:04d}'.format(data_idx, data_idx_range[0], data_idx_range[-1]), end='\r')

        if is_use_from_real:
            if data_idx >= 150:
                data_idx += 50
                src_frame_dir = src_2_dir + '/proj'
                src_gt_dir = src_2_dir + '/gt'
                src_shading_dir = src_2_dir + '/shade'
                src_rec_dir = src_2_dir + '/rec'

        if is_transfer_learning or is_finetune or is_transfer_encoder:
            src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
            # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(data_idx)
            src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
            src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
            # src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)
            src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)
            if (data_idx >= 16 and data_idx <= 20) or data_idx >= 25:
                src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)
        else:
            src_bgra = src_frame_dir + '/{:05d}.png'.format(data_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(data_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(data_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(data_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1)
        bgr = bgr[:img_size, :img_size, :] / 255.
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        depth_img_gap = depth_img_gap[:img_size, :img_size, :]
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_img_gt = depth_img_gt[:img_size, :img_size, :]
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        # shading_bgr = cv2.imread(src_shading, -1)
        # shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]
        shading_gray = cv2.imread(src_shading, 0) # GrayScale
        shading = shading_gray[:img_size, :img_size]

        # is_shading_available = shading > 0
        is_shading_available = shading > 16.0
        mask_shading = is_shading_available * 1.0
        # depth_gap = depth_gt[:, :] * mask_shading
        # mean_depth = np.sum(depth_gap) / np.sum(mask_shading)
        # depth_gap = mean_depth * mask_shading
        depth_gap *= mask_shading

        # Noise
        if is_noise_shading:
            noise = np.zeros((img_size, img_size))
            for i_perlin in np.random.rand(freq_num) * freq_range:
                i_perlin += freq_range
                perlin_linspace = np.linspace(0, 8*i_perlin, img_size)
                perlin_meshgrid = np.array(np.meshgrid(perlin_linspace, perlin_linspace))
                noise += perlin(perlin_meshgrid, seed=data_idx)
                noise *= noise_intensity
            shading += noise.astype('uint8')
            shading *= mask_shading.astype('uint8')
        if is_noise_gt:
            h_gauss, w_gauss = depth_gt.shape
            gauss = np.random.normal(noise_mean, noise_sigma, (h_gauss, w_gauss))
            gauss = gauss.reshape(h_gauss, w_gauss)
            depth_gt += gauss

        if is_shading_norm: # shading norm : mean 0, var 1
            # is_shading_available = shading > 16.0
            # mask_shading = is_shading_available * 1.0
            mean_shading = np.sum(shading*mask_shading) / np.sum(mask_shading)
            var_shading = np.sum(np.square((shading - mean_shading)*mask_shading)) / np.sum(mask_shading)
            std_shading = np.sqrt(var_shading)
            shading = (shading - mean_shading) / std_shading
        else:
            shading = shading / 255.

        # is_depth_available = depth_gt > depth_threshold
        # mask_depth = is_depth_available * 1.0
        # depth_gap = np.zeros_like(depth_gt)
        # mean_depth = np.sum(depth_gt) / np.sum(mask_depth)
        # depth_gap = mean_depth * mask_depth


        # normalization (may not be needed)
        # depth_gap /= depth_gap.max()
        # depth_gt /= depth_gt.max()

        if is_input_coord:
            coord_x = np.linspace(0, 1, img_size)
            coord_y = np.linspace(0, 1, img_size)
            grid_x, grid_y = np.meshgrid(coord_x, coord_y)

        # merge bgr + depth_gap
        if is_input_frame:
            if is_input_depth:
                bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0]])
                if is_input_coord:
                    bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0], grid_x, grid_y])
            else:
                bgrd = np.dstack([shading[:, :], bgr[:, :, 0]])
        else:
            bgrd = np.dstack([shading[:, :], depth_gap])

        # difference
        difference = depth_gt - depth_gap
        # mask
        is_gap_available = depth_gap > depth_threshold
        is_depth_close = np.logical_and(
                np.abs(difference) < difference_threshold,
                is_gap_available)
        mask = is_depth_close.astype(np.float32)
        length = np.sum(mask)

        # mean_difference = np.sum(difference * mask) / length
        # difference = (difference - mean_difference) * mask

        if is_difference_norm:
            mean_difference = np.sum(difference * mask) / length
            var_difference = np.sum(np.square((difference - mean_difference)*mask)) / length
            std_difference = np.sqrt(var_difference)
            difference = (difference - mean_difference) / std_difference

        # gt = np.dstack([difference, mask])
        # gt = np.dstack([difference])
        gt = np.dstack([difference, depth_gap])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(b_top, img_shape[0], b_h)
        left_coords = range(b_left, img_shape[1], b_w)

        # add training data
        for top, left in product(top_coords, left_coords):
            batch_train = clip_batch(bgrd, (top, left), batch_shape)
            batch_gt = clip_batch(gt, (top, left), batch_shape)
            batch_mask = clip_batch(mask, (top, left), batch_shape)

            # batch_mask = batch_gt[:, :, 1]

            # do not add batch if not close ################
            if np.sum(batch_mask) < (b_h * b_w * patch_remove):
                valid.append(False)
                continue
            else:
                valid.append(True)

            if not return_valid:
                if is_input_depth or is_input_frame:
                    x_train.append(batch_train)
                else:
                    # x_train.append(batch_train[:, :, 0].reshape((*batch_shape, 1)))
                    x_train.append(batch_train[:, :, 0].reshape((batch_shape[0], batch_shape[1], 1)))
                # y_train.append(batch_gt.reshape((*batch_shape, 2)))
                y_train.append(batch_gt)

    if return_valid:
        print('\n')
        return valid
    else:
        return np.array(x_train), np.array(y_train)


def prepare_batch(batch_idx_start, batch_idx_end, list_valid):
    def clip_batch(img, top_left, size):
        t, l, h, w = *top_left, *size
        return img[t:t + h, l:l + w]

    if is_transfer_learning or is_finetune:
        src_rec_dir = src_dir + '/rec'
        src_rec_dir = src_dir + '/rec_ajusted'
        src_frame_dir = src_dir + '/frame'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shading'
    else:
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'

    # read data
    # print('loading data...')
    img_idx_start = int(batch_idx_start/100)
    img_idx_end = int(batch_idx_end/100) + 1
    data_idx_range = range(img_idx_start, img_idx_end)
    x_train = []
    y_train = []
    # for data_idx in tqdm(data_idx_range):
    for data_idx in data_idx_range:
        if is_transfer_learning or is_finetune:
            src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
            # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(data_idx)
            src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
            src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
            # src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)
            src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)
        else:
            src_bgra = src_frame_dir + '/{:05d}.png'.format(data_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(data_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(data_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(data_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        bgr = bgr[:1200, :1200, :]
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_img_gap = depth_img_gap[:1200, :1200, :]
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_img_gt = depth_img_gt[:1200, :1200, :]
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        # shading_bgr = cv2.imread(src_shading, -1)
        # shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]
        shading_gray = cv2.imread(src_shading, 0) # GrayScale
        shading_gray = shading_gray[:1200, :1200]
        shading = shading_gray

        is_shading_available = shading > 0
        mask_shading = is_shading_available * 1.0
        # depth_gap = depth_gt[:, :] * mask_shading
        # mean_depth = np.sum(depth_gap) / np.sum(mask_shading)
        # depth_gap = mean_depth * mask_shading
        depth_gap *= mask_shading

        if is_shading_norm: # shading norm : mean 0, var 1
            is_shading_available = shading > 16.0
            mask_shading = is_shading_available * 1.0
            mean_shading = np.sum(shading*mask_shading) / np.sum(mask_shading)
            var_shading = np.sum(np.square((shading - mean_shading)*mask_shading)) / np.sum(mask_shading)
            std_shading = np.sqrt(var_shading)
            shading = (shading - mean_shading) / std_shading
        else:
            shading = shading / 255.

        # is_depth_available = depth_gt > depth_threshold
        # mask_depth = is_depth_available * 1.0
        # depth_gap = np.zeros_like(depth_gt)
        # mean_depth = np.sum(depth_gt) / np.sum(mask_depth)
        # depth_gap = mean_depth * mask_depth


        # normalization (may not be needed)
        # depth_gap /= depth_gap.max()
        # depth_gt /= depth_gt.max()


        # merge bgr + depth_gap
        if is_input_frame:
            if is_input_depth:
                bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0]])
            else:
                bgrd = np.dstack([shading[:, :], bgr[:, :, 0]])
        else:
            bgrd = np.dstack([shading[:, :], depth_gap])

        # difference
        difference = depth_gt - depth_gap
        # mask
        is_gap_available = depth_gap > depth_threshold
        is_depth_close = np.logical_and(
                np.abs(difference) < difference_threshold,
                is_gap_available)
        mask = is_depth_close.astype(np.float32)
        length = np.sum(mask)

        # mean_difference = np.sum(difference * mask) / length
        # difference = (difference - mean_difference) * mask

        if is_difference_norm:
            mean_difference = np.sum(difference * mask) / length
            var_difference = np.sum(np.square((difference - mean_difference)*mask)) / length
            std_difference = np.sqrt(var_difference)
            difference = (difference - mean_difference) / std_difference

        # gt = np.dstack([difference, mask])
        # gt = np.dstack([difference])
        gt = np.dstack([difference, depth_gap])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(b_top, img_shape[0], b_h)
        left_coords = range(b_left, img_shape[1], b_w)

        # add training data
        cnt_valid = -1
        cnt_img = img_idx_start*100 - 1
        for top, left in product(top_coords, left_coords):
            batch_train = clip_batch(bgrd, (top, left), batch_shape)
            batch_gt = clip_batch(gt, (top, left), batch_shape)
            batch_mask = clip_batch(mask, (top, left), batch_shape)

            # batch_mask = batch_gt[:, :, 1]

            # do not add batch if not close ################
            cnt_valid += 1
            cnt_img += 1
            if (cnt_img < batch_idx_start) or (cnt_img > batch_idx_end):
                continue
            if not list_valid[cnt_valid]:
                continue

            if is_input_depth or is_input_frame:
                x_train.append(batch_train)
            else:
                # x_train.append(batch_train[:, :, 0].reshape((*batch_shape, 1)))
                x_train.append(batch_train[:, :, 0].reshape((batch_shape[0], batch_shape[1], 1)))
            # y_train.append(batch_gt.reshape((*batch_shape, 2)))
            y_train.append(batch_gt)

    return np.array(x_train), np.array(y_train)


# class BatchGenerator(Sequence):
#     def __init__(self, data_range, batch_size=32):
#         self.valid = prepare_data(data_range, return_valid=True)
#         self.length = np.sum(self.valid * 1)
#         self.batch_size = batch_size
#         # self.batches_per_epoch = int((self.length - 1) / batch_size) + 1
#         self.batches_per_epoch = int((self.length - 1) / batch_size)
#         self.list_valid_idx = []
#         for i, vali in enumerate(self.valid):
#             if vali:
#                 self.list_valid_idx.append(i)
#         # print(self.list_valid_idx)

#     def __getitem__(self, idx):
#         batch_from = self.list_valid_idx[self.batch_size * idx]
#         batch_to = self.list_valid_idx[self.batch_size * (idx + 1) - 1]

#         return prepare_batch(batch_from, batch_to, self.valid[int(batch_from/100)*100: int(batch_to/100 + 1)*100])

#     def __len__(self):
#         return self.batches_per_epoch

#     def on_epoch_end(self):
#         pass

class BatchGenerator(Sequence):
    def __init__(self, dir_name, data_num):
        self.batches_per_epoch = data_num
        self.input_dir = dir_name + '/in'
        self.gt_dir = dir_name + '/gt'

    def __getitem__(self, idx):
        input_batch = np.load(self.input_dir + '/{:05d}.npy'.format(idx))
        gt_batch = np.load(self.gt_dir + '/{:05d}.npy'.format(idx))
        return input_batch, gt_batch

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

is_aug_lumi = True
# is_aug_lumi = False

class MiniBatchGenerator(Sequence):
    def __init__(self, dir_name, data_num, use_num):
        self.data_size = data_num
        self.batches_per_epoch = use_num
        self.x_file = dir_name + '/x/{:05d}.npy'
        self.y_file = dir_name + '/y/{:05d}.npy'

    def __getitem__(self, idx):
        random_idx = random.randrange(0, self.data_size)
        x_batch = np.load(self.x_file.format(random_idx))
        y_batch = np.load(self.y_file.format(random_idx))
        if is_aug_lumi:
            aug_scale = random.uniform(lumi_scale_range[0], lumi_scale_range[1])
            x_batch[:, :, :, 0] *= aug_scale
            x_batch[:, :, :, 2] *= aug_scale
        return x_batch, y_batch

    def __len__(self):
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

def main():
    if is_model_exist is '0':
        os.makedirs(out_dir)

    if use_generator:
        # patch_dir = '../data/patch_wave1'
        # train_generator = BatchGenerator(patch_dir + '/train', 66)
        # val_generator = BatchGenerator(patch_dir + '/val', 29)

        # patch_dir = '../data/patch_wave2_1000'
        # train_generator = BatchGenerator(patch_dir + '/train', 328)
        # val_generator = BatchGenerator(patch_dir + '/val', 144)

        # patch_dir = '../data/patch_wave2_2000'
        # train_generator = BatchGenerator(patch_dir + '/train', 656)
        # val_generator = BatchGenerator(patch_dir + '/val', 289)

        # wave1
        #100
        patch_dir = '../data/batch_wave1_100'
        train_generator = MiniBatchGenerator(patch_dir + '/train', 46, 70)
        val_generator = MiniBatchGenerator(patch_dir + '/val', 20, 30)
        #400
        # patch_dir = '../data/batch_wave1_400'
        # train_generator = MiniBatchGenerator(patch_dir + '/train', 197, 70)
        # val_generator = MiniBatchGenerator(patch_dir + '/val', 80, 30)

        # wave1-double
        # 200
        # patch_dir = '../data/batch_wave1-double_200'
        # train_generator = MiniBatchGenerator(patch_dir + '/train', 94, 70)
        # val_generator = MiniBatchGenerator(patch_dir + '/val', 40, 30)
        # 800
        # patch_dir = '../data/batch_wave1-double_800'
        # train_generator = MiniBatchGenerator(patch_dir + '/train', 395, 70)
        # val_generator = MiniBatchGenerator(patch_dir + '/val', 168, 30)
        
    else:
        x_data, y_data = prepare_data(data_idx_range)
        print('x train data:', x_data.shape)
        print('y train data:', y_data.shape)
        if is_augment:
            if is_transfer_learning or is_finetune or is_transfer_encoder:
                x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                                test_size=val_rate, shuffle=True)
            else:
                # x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                #                                                 test_size=val_rate, shuffle=False)
                x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                                test_size=val_rate, shuffle=True)
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
            if is_input_coord:
                ch_num = 5
        else:
            ch_num = 2
    else:
        if is_input_frame:
            ch_num = 2
        else:
            ch_num = 1

    # model configuration
    if net_type is '0':
        model = network.build_unet_model(
            batch_shape,
            ch_num,
            # decay=decay,
            drop_rate=dropout_rate,
            transfer_learn=is_transfer_learning,
            transfer_encoder=is_transfer_encoder,
            lr=learning_rate,
            scaling=difference_scaling
            )
    elif net_type is '1':
        model = network.build_resnet_model(
            batch_shape,
            ch_num,
            depth_threshold=depth_threshold,
            difference_threshold=difference_threshold,
            drop_rate=dropout_rate,
            scaling=difference_scaling
            )
    elif net_type is '2':
        model = network.build_dense_resnet_model(
            batch_shape,
            ch_num,
            depth_threshold=depth_threshold,
            difference_threshold=difference_threshold,
            drop_rate=dropout_rate
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

    elif resume_from == 'from_min':
        model_file = model_dir + '/model-%03d.hdf5'%idx_min_loss
        initial_epoch = pre_epoch

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
        # os.makedirs(out_dir, exist_ok=True)
        # os.makedirs(model_dir, exist_ok=True)
        # os.makedirs(out_dir)
        os.makedirs(model_dir)

    # train
    # model_save_cb = ModelCheckpoint(model_dir + '/model-best.hdf5',
    #                                 # model_dir + '/model-{epoch:03d}.hdf5',
    #                                 monitor=monitor_loss,
    #                                 verbose=verbose,
    #                                 save_best_only=True,
    #                                 save_weights_only=True,
    #                                 mode='min',
    #                                 period=1)
    model_save_cb = ModelCheckpoint(model_dir + '/model-{epoch:03d}.hdf5',
                                    period=save_period,
                                    save_weights_only=True)
    csv_logger_cb = CSVLogger(out_dir + '/training.log',
                              append=(resume_from is not None))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.5,
                                    patience=10,
                                    verbose=1)

    print('training')
    if use_generator:
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.batches_per_epoch,
            epochs=epoch_num,
            initial_epoch=initial_epoch,
            shuffle=True,
            callbacks=[model_save_cb, csv_logger_cb],
            validation_data=val_generator,
            validation_steps=val_generator.batches_per_epoch,
            verbose=verbose,
            max_queue_size=2)
    else:
        if is_augment:
            model.fit_generator(
                train_generator,
                steps_per_epoch=len(x_train)*augment_rate / train_batch_size + 1,
                epochs=epoch_num,
                initial_epoch=initial_epoch,
                shuffle=True,
                callbacks=[model_save_cb, csv_logger_cb],
                # callbacks=[model_save_cb, csv_logger_cb, reduce_lr],
                validation_data=val_generator,
                validation_steps=len(x_val)*augment_val_rate / train_batch_size + 1,
                verbose=verbose)
        else:
            model.fit(
                x_train,
                y_train,
                epochs=epoch_num,
                batch_size=train_batch_size,
                initial_epoch=initial_epoch,
                shuffle=True,
                validation_split=val_rate,
                callbacks=[model_save_cb, csv_logger_cb],
                verbose=verbose)

    model.save_weights(out_dir + '/model-final.hdf5')
    # model.save_weights(model_dir + '/model-final.hdf5')


if __name__ == "__main__":
    main()
