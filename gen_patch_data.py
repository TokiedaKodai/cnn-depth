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

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set currenct dir

# src_dir = '../data/render_wave1'
src_dir = '../data/render_wave2_1000'
data_num = 1000

# save_dir = '../data/patch_wave1'
save_dir = '../data/patch_wave2_2000'

# parameters
depth_threshold = 0.2
difference_threshold = 0.01
patch_remove = 0.9

is_transfer_learning = False
is_finetune = False

# input
is_input_depth = True
is_input_frame = True

# normalization
is_shading_norm = True # Shading Normalization
# is_shading_norm = False
is_difference_norm = True # Difference Normalization
# is_difference_norm = False

batch_shape = (120, 120)
batch_tl = (0, 0)  # top, left

train_batch_size = 64

# val_rate = 0.1
val_rate = 0.3



def gen_patch_data(data_idx_range, dir_name, batch_size=64):
    input_dir = dir_name + '/in'
    gt_dir = dir_name + '/gt'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

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
    data_idx_range = list(data_idx_range)
    x_train = []
    y_train = []
    valid = []
    len_list = 0
    num_patch = 0
    num_patch = 328
    num_patch = 144
    for data_idx in tqdm(data_idx_range):
    # for data_idx in data_idx_range:
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
        depth_img_gap = depth_img_gap[:1200, :1200, :]
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_img_gt = depth_img_gt[:1200, :1200, :]
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        # shading_bgr = cv2.imread(src_shading, -1)
        # shading_bgr = shading_bgr[:1200, :1200, :]
        # shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]
        shading_gray = cv2.imread(src_shading, 0) # GrayScale
        shading_gray = shading_gray[:1200, :1200]
        shading = shading_gray#.reshape(shading_gray.shape + (1,))

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

            len_list += 1
            if is_input_depth or is_input_frame:
                x_train.append(batch_train)
            else:
                # x_train.append(batch_train[:, :, 0].reshape((*batch_shape, 1)))
                x_train.append(batch_train[:, :, 0].reshape((batch_shape[0], batch_shape[1], 1)))
            # y_train.append(batch_gt.reshape((*batch_shape, 2)))
            y_train.append(batch_gt)

            if (len_list % batch_size) == 0:
                np.save(input_dir + '/{:05d}.npy'.format(num_patch), np.array(x_train))
                np.save(gt_dir + '/{:05d}.npy'.format(num_patch), np.array(y_train))
                x_train = []
                y_train = []
                num_patch += 1


def main():
    train_dir = save_dir + '/train'
    val_dir = save_dir + '/val'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_num = int(data_num * (1 - val_rate))

    # gen_patch_data(range(train_num), train_dir)
    gen_patch_data(range(train_num, data_num), val_dir)


if __name__ == "__main__":
    main()