import cv2
import numpy as np
from itertools import product
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import io
import sys
import pandas as pd

import network
import depth_tools
import common_tools
import compare_error

'''
ARGV
1: output dir
2: epoch num
3: is predict norm
'''
argv = sys.argv
# _, out_dir, epoch_num, net_type = argv # output dir, epoch, network type
# _, out_dir, epoch_num, data_type = argv # output dir, epoch
_, out_dir, epoch_num = argv

net_type = '0'

out_dir = '../output/output_' + out_dir
# out_dir = '../output/archive/200318/output_' + out_dir

epoch_num = int(epoch_num)

# normalization
is_shading_norm = True
# is_shading_norm = False
is_difference_norm = True
# is_difference_norm = False

# parameters
depth_threshold = 0.2
# difference_threshold = 0.1
difference_threshold = 0.005
difference_threshold = 0.01
patch_remove = 0.5

# input
is_input_depth = True
is_input_frame = True

#select model
# save_period = 10
save_period = 1

# select_range = save_period * 10
select_range = epoch_num


mask_edge_size = 4
mask_edge_size = 2

#InputData
src_dir = '../data/render'
data_num = 160

# predict normalization
is_predict_norm = True
is_predict_norm = False

is_pred_ajust = True
is_pred_ajust = False

# select from val loss
is_select_val = True
is_select_val = False

# Reverse #############################
is_pred_reverse = True
is_pred_reverse = False

is_pred_pix_reverse = True
is_pred_pix_reverse = False

is_reverse_threshold = True
is_reverse_threshold = False

r_thre = 0.002
#######################################

is_pred_smooth = True
is_pred_smooth = False


data_idx_range = list(range(data_num))


vmin, vmax = (0.8, 1.4)
vm_range = 0.05
# vm_e_range = 0.005
vm_e_range = 0.003
# vm_e_range = difference_threshold

batch_shape = (1200, 1200)
# batch_shape = (600, 600)
batch_tl = (0, 0)  # top, left

# val_rate = 0.1
val_rate = 0.3

# Train, Val
train_num = int(data_num * (1 - val_rate))
train_range = range(train_num)
val_range = range(train_num, data_num)


train_std = 0.0019195375434992092

# difference_scaling = 100
difference_scaling = 1
# difference_scaling = 1 / train_std

def prepare_data(data_idx_range):
    src_frame_dir = src_dir + '/proj'
    src_gt_dir = src_dir + '/gt'
    src_shading_dir = src_dir + '/shade'
    src_rec_dir = src_dir + '/rec'

    # read data
    print('loading data...')
    img_x = []
    img_y = []
    for data_idx in tqdm(data_idx_range):
        src_bgra = src_frame_dir + '/{:05d}.png'.format(data_idx)
        src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(data_idx)
        src_shading = src_shading_dir + '/{:05d}.png'.format(data_idx)
        src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(data_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        # shading_bgr = cv2.imread(src_shading, -1)
        # shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]
        shading_gray = cv2.imread(src_shading, 0) # GrayScale
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

        depth_thre = depth_threshold

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
        # difference *= difference_scaling # difference scaling

        # mask
        is_gt_available = depth_gt > depth_thre
        is_depth_close = np.logical_and(
                np.abs(difference) < difference_threshold,
                is_gt_available)
        mask = is_depth_close.astype(np.float32)
        length = np.sum(mask)

        # mean_difference = np.sum(difference * mask) / length
        # difference = (difference - mean_difference) * mask

        if is_difference_norm:
            mean_difference = np.sum(difference * mask) / length
            var_difference = np.sum(np.square((difference - mean_difference)*mask)) / length
            std_difference = np.sqrt(var_difference)
            difference = (difference - mean_difference) / std_difference

        gt = np.dstack([difference, mask])

        # # clip batches
        # b_top, b_left = batch_tl
        # b_h, b_w = batch_shape
        # top_coords = range(b_top, img_shape[0], b_h)
        # left_coords = range(b_left, img_shape[1], b_w)
        # # add test data
        # x_test = []
        # for top, left in product(top_coords, left_coords):

        #     def clip_batch(img, top_left, size):
        #         t, l, h, w = *top_left, *size
        #         return img[t:t + h, l:l + w]

        #     batch_test = clip_batch(bgrd, (top, left), batch_shape)

        #     if is_input_depth or is_input_frame:
        #         x_test.append(batch_test)
        #     else:
        #         x_test.append(batch_test[:, :, 0].reshape((*batch_shape, 1)))

        # img_x.append(np.array(x_test)[:])
        # img_x.append(x_test)
        img_x.append(bgrd[:1200, :1200, :])
        img_y.append(gt[:1200, :1200, :])
    return np.array(img_x), np.array(img_y), depth_thre

def main():
    x_data, y_data, depth_thre = prepare_data(data_idx_range)
    # print('x train data:', x_data.shape)
    # print('y train data:', y_data.shape)

    # x_train  = x_data[0: train_num]
    # x_val = x_data[train_num: data_num]
    # y_train = y_data[0: train_num]
    # y_val = y_data[train_num: data_num]

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
    if net_type is '0':
        model = network.build_unet_model(batch_shape, ch_num)
    elif net_type is '1':
        model = network.build_resnet_model(batch_shape, ch_num)
    

    loss_str = 'epoch,loss,val_loss\n'

    for epoch in tqdm(range(1, epoch_num + 1)):
        model.load_weights(out_dir + '/model/model-%03d.hdf5'%epoch)
        loss_str += str(epoch) + ','

        # Train Loss
        list_err = []
        list_length = []
        for idx in train_range:
            # x = x_data[idx]
            x = x_data[idx].reshape((1, 1200, 1200, 3))
            y = y_data[idx]

            predict = model.predict(x, batch_size=1)
            decode_img = predict[0][:, :, 0:2]

            pred_diff = decode_img[:, :, 0].copy()
            gt = y[:, :, 0]
            mask = y[:, :, 1]

            mask_length = np.sum(mask)
            err = np.sum(np.square(gt - pred_diff) * mask)

            list_err.append(err)
            list_length.append(mask_length)

        mse = np.sum(list_err) / np.sum(list_length)
        # rmse = np.sqrt(mse)

        loss_str += str(mse) + ','

        # Val Loss
        list_err = []
        list_length = []
        for idx in val_range:
            # x = x_data[idx]
            x = x_data[idx].reshape((1, 1200, 1200, 3))
            y = y_data[idx]

            predict = model.predict(x, batch_size=1)
            decode_img = predict[0][:, :, 0:2]

            pred_diff = decode_img[:, :, 0].copy()
            gt = y[:, :, 0]
            mask = y[:, :, 1]

            mask_length = np.sum(mask)
            err = np.sum(np.square(gt - pred_diff) * mask)

            list_err.append(err)
            list_length.append(mask_length)

        mse = np.sum(list_err) / np.sum(list_length)
        # rmse = np.sqrt(mse)

        loss_str += str(mse) + '\n'

    with open(out_dir + '/loss.txt', mode='w') as f:
        f.write(loss_str)

if __name__ == "__main__":
    main()
