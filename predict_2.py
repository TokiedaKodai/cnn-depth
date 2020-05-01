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
_, out_dir, epoch_num, data_type = argv # output dir, epoch

net_type = '0'

out_dir = '../output/output_' + out_dir
# out_dir = '../output/archive/200318/output_' + out_dir

epoch_num = int(epoch_num)

# normalization
is_shading_norm = True
# is_shading_norm = False

# parameters
depth_threshold = 0.2
# difference_threshold = 0.1
difference_threshold = 0.005
# difference_threshold = 0.003
patch_remove = 0.5
difference_scaling = 100
# difference_scaling = 10

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
# src_dir = '../data/input_200201'
# src_dir = '../data/input_200302'
# src_dir = '../data/input_200201-0312/'

if data_type is '0':
    src_dir = '../data/input_200318'
    predict_dir = out_dir + '/predict_{}_board'.format(epoch_num)
    # predict_dir = out_dir + '/predict_{}_trans'.format(epoch_num)
    data_num = 68
elif data_type is '1':
    src_dir = '../data/render'
    # src_dir = '../data/render_no-tilt'
    predict_dir = out_dir + '/predict_{}'.format(epoch_num)
    # data_num = 100
    data_num = 200
elif data_type is '2':
    src_dir = '../data/real'
    predict_dir = out_dir + '/predict_{}_real'.format(epoch_num)
    data_num = 9


# save predict depth PLY file
is_save_ply = True
# is_save_ply = False

is_masked_ply = True

is_save_diff = True
is_save_diff = False

# predict normalization
is_predict_norm = True
# is_predict_norm = False

is_pred_ajust = True
is_pred_ajust = False

# select from val loss
is_select_val = True
# is_select_val = False

is_pred_reverse = True
# is_pred_reverse = False

is_pred_pix_reverse = True
# is_pred_pix_reverse = False

is_pred_smooth = True
# is_pred_smooth = False

if is_predict_norm:
    predict_dir += '_norm'
if is_pred_ajust:
    predict_dir += '_ajust'
if is_pred_smooth:
    predict_dir += '_smooth'
if is_pred_reverse:
    predict_dir += '_reverse'
if is_pred_pix_reverse:
    predict_dir += '_pix'

data_idx_range = list(range(data_num))

'''
Test Data
ori 110cm : 16 - 23
small 100cm : 44 - 47
mid 110cm : 56 - 59
'''
# test data
if data_type is '0':
    test_range = list(range(16, 24)) # 16 - 24
    test_range.extend(list(range(44, 48))) # 44 - 48
    test_range.extend(list(range(56, 60))) # 56 - 60
elif data_type is '1':
    # test_range = list(range(80, 100))
    test_range = list(range(160, 200))
elif data_type is '2':
    test_range = list(range(9))

# train data
if data_type is '0':
    train_range = list(range(16)) # 0 - 16, 24 - 40
    train_range.extend(list(range(24, 40)))
    train_range.extend(list(range(40, 44))) # 40 - 44
    train_range.extend(list(range(48, 56))) # 48 - 56, 60 - 68
    train_range.extend(list(range(60, 68)))
elif data_type is '1':
    train_range = list(range(160))
elif data_type is '2':
    train_range = list()

# save ply range
save_ply_range = test_range

save_img_range = test_range


vmin, vmax = (0.8, 1.4)
vm_range = 0.05
# vm_e_range = 0.005
vm_e_range = difference_threshold

batch_shape = (1200, 1200)
# batch_shape = (600, 600)
batch_tl = (0, 0)  # top, left

# calib 200317
cam_params = {
    'focal_length': 0.037009,
    'pix_x': 1.25e-05,
    'pix_y': 1.2381443057539635e-05,
    'center_x': 790.902,
    'center_y': 600.635
}
# calib 200427
# cam_params = {
#     'focal_length': 0.036917875,
#     'pix_x': 1.25e-05,
#     'pix_y': 1.2416172558410155e-05,
#     'center_x': 785.81,
#     'center_y': 571.109
# }

def main():
    if data_type is '0':
        src_rec_dir = src_dir + '/rec'
        src_rec_dir = src_dir + '/rec_ajusted'
        src_frame_dir = src_dir + '/frame'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shading'
    elif data_type is '1':
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'
    elif data_type is '2':
        src_frame_dir = src_dir + '/proj'
        src_gt_dir = src_dir + '/gt'
        src_shading_dir = src_dir + '/shade'
        src_rec_dir = src_dir + '/rec'

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
    elif net_type is '2':
        model = network.build_dense_resnet_model(batch_shape, ch_num)
    
    # log
    df_log = pd.read_csv(out_dir + '/training.log')
    if is_select_val:
        df = df_log['val_loss']
    else:
        df = df_log['loss']
    
    df.index = df.index + 1
    # select minimum loss model
    is_select_min_loss_model = True
    if is_select_min_loss_model:
        df_select = df[df.index>epoch_num-select_range]
        df_select = df_select[df_select.index<=epoch_num]
        df_select = df_select[df_select.index%save_period==0]
        min_loss = df_select.min()
        idx_min_loss = df_select.idxmin()
        # model.load_weights(out_dir + '/model/model-%03d.hdf5'%idx_min_loss)
        model.load_weights(out_dir + '/model/model-best.hdf5')
    else:
        model.load_weights(out_dir + '/model-final.hdf5')
    
    # loss graph
    lossgraph_dir = predict_dir + '/loss_graph'
    os.makedirs(lossgraph_dir, exist_ok=True)
    arr_loss = df.values
    arr_epoch = df.index
    if is_select_val:
        init_epochs = [0, 10, int(epoch_num / 2), epoch_num - 200]
    else:
        init_epochs = [0, 10, epoch_num - 200, epoch_num - 100]

    for init_epo in init_epochs:
        if init_epo < 0:
            continue
        if init_epo >= epoch_num:
            continue
        plt.figure()
        plt.plot(arr_epoch[init_epo: epoch_num], arr_loss[init_epo: epoch_num])
        if is_select_min_loss_model:
            plt.plot(idx_min_loss, min_loss, 'ro')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('{} : epoch {} - {}'.format(df.name, init_epo + 1, epoch_num))
        plt.savefig(lossgraph_dir + '/loss_{}-{}.pdf'.format(init_epo + 1, epoch_num))

    # error compare txt
    err_strings = 'index,type,MAE depth,MAE predict,RMSE depth,RMSE predict\n'

    os.makedirs(predict_dir, exist_ok=True)
    for test_idx in tqdm(test_range):
        if data_type is '0':
            src_bgra = src_frame_dir + '/frame{:03d}.png'.format(test_idx)
            # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(test_idx)
            src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(test_idx)
            # src_shading = src_shading_dir + '/shading{:03d}.png'.format(test_idx)
            src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(test_idx)
        elif data_type is '1':
            src_bgra = src_frame_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(test_idx)
        elif data_type is '2':
            src_bgra = src_frame_dir + '/{:05d}.png'.format(test_idx)
            src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(test_idx)
            src_shading = src_shading_dir + '/{:05d}.bmp'.format(test_idx)
            src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(test_idx)

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
        shading_bgr = cv2.imread(src_shading, 1)
        shading_bgr = shading_bgr[:1200, :1200, :]
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
        # norm_factor = depth_gap.max()
        # depth_gap /= norm_factor
        # depth_gt /= depth_gt.max()

        depth_thre = depth_threshold

        # merge bgr + depth_gap
        if is_input_frame:
            if is_input_depth:
                bgrd = np.dstack([shading[:, :], depth_gap, bgr[:, :, 0]])
            else:
                bgrd = np.dstack([shading[:, :], bgr[:, :, 0]])
        else:
            bgrd = np.dstack([shading[:, :], depth_gap])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(b_top, img_shape[0], b_h)
        left_coords = range(b_left, img_shape[1], b_w)

        # add training data
        x_test = []
        for top, left in product(top_coords, left_coords):

            def clip_batch(img, top_left, size):
                t, l, h, w = *top_left, *size
                return img[t:t + h, l:l + w]

            batch_test = clip_batch(bgrd, (top, left), batch_shape)
            
            if is_input_depth or is_input_frame:
                x_test.append(batch_test)
            else:
                x_test.append(batch_test[:, :, 0].reshape((*batch_shape, 1)))

        # predict
        x_test = np.array(x_test)[:]
        predict = model.predict(x_test, batch_size=1)  # w/o denormalization
        # predict = model.predict(
        #     x_test, batch_size=1) * norm_factor  # w/ denormalization
        # decode_img = np.hstack([
        #     np.vstack([predict[0], predict[2]]),
        #     np.vstack([predict[1], predict[3]])
        # ])[:, :, 0:2]
        decode_img = predict[0][:, :, 0:2]

        # training types
        is_gt_available = depth_gt > depth_thre
        is_gap_unavailable = depth_gap < depth_thre

        is_depth_close = np.logical_and(
            np.abs(depth_gap - depth_gt) < difference_threshold,
            is_gt_available)

        is_to_interpolate = np.logical_and(is_gt_available, is_gap_unavailable)
        train_segment = np.zeros(decode_img.shape[:2])
        train_segment[is_depth_close] = 1
        train_segment[is_to_interpolate] = 2

        # is_train_area = np.logical_or(is_depth_close, is_to_interpolate)

        # mask = is_gt_available * 1.0 # GT
        mask = is_depth_close * 1.0 # no-complement
        # mask = is_train_area * 1.0 # complement

        # delete mask edge
        edge_size = mask_edge_size
        mask_filter = np.zeros_like(mask)
        for edge in range(1, edge_size):
            edge_2 = edge * 2
            mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[: b_h - edge_2, edge: b_w - edge]
            mask *= mask_filter
            mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge: b_h - edge, edge_2: ]
            mask *= mask_filter
            mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge_2: , edge: b_w - edge]
            mask *= mask_filter
            mask_filter[edge: b_h - edge, edge: b_w - edge] = mask[edge: b_h - edge, : b_w - edge_2]
            mask *= mask_filter

            for i in range(2):
                for j in range(2):
                    mask_filter[
                        edge * i: b_h - edge * (1 - i), edge * j: b_w - edge * (1 - j)
                        ] = mask[
                            edge * (1 - i): b_h - edge * i, edge * (1 - j): b_h - edge * j
                            ]
                    mask *= mask_filter


        mask_gt = is_gt_available * 1.0

        mask_length = np.sum(mask)

        # cv2.imwrite(predict_dir + '/mask-{:05d}.png'.format(test_idx),
        #             (mask * 255).astype(np.uint8))

        predict_depth = decode_img[:, :, 0].copy()

        if is_pred_smooth:
            predict_depth = common_tools.gaussian_filter(predict_depth, 2, 0.002)

        depth_gt_masked = depth_gt * mask
        gt_diff = (depth_gt - depth_gap) * mask
        predict_masked = predict_depth * mask

        # scale
        predict_depth /= difference_scaling
        predict_masked /= difference_scaling

        # reverse predict
        if is_pred_reverse:
            # predict normalization
            if is_predict_norm:
                mean_gt = np.sum(gt_diff) / mask_length
                mean_predict = np.sum(predict_masked) / mask_length
                gt_diff -= mean_gt
                predict_depth -= mean_predict
                predict_depth *= -1.0 # reverse
                out_diff_R = predict_depth.copy() # save diff
                sd_gt = np.sqrt(np.sum(np.square((gt_diff)*mask)) / mask_length)
                sd_predict = np.sqrt(np.sum(np.square((predict_depth)*mask)) / mask_length)
                predict_depth *= sd_gt / sd_predict
                predict_depth += mean_gt
                predict_masked = predict_depth * mask

            # difference learn
            predict_depth += depth_gap
            predict_masked += depth_gap * mask

            # ajust bias, calc error
            if is_pred_ajust:
                mean_gt_diff = np.sum(gt_diff) / mask_length
                mean_out_dif = np.sum(out_diff * mask) / mask_length
                bias_out_dif = mean_out_dif - mean_gt_diff
                out_diff_ajusted = out_diff - bias_out_dif

            # error
            depth_err_abs_R = np.abs(depth_gt - depth_gap)
            depth_err_sqr_R = np.square(depth_gt - depth_gap)
            if is_pred_ajust:
                predict_err_abs_R = np.abs(gt_diff - out_diff_ajusted)
                predict_err_sqr_R = np.square(gt_diff - out_diff_ajusted)
            else:
                predict_err_abs_R = np.abs(depth_gt - predict_depth)
                predict_err_sqr_R = np.square(depth_gt - predict_depth)

            # error image
            depth_err_R = depth_err_abs_R
            predict_err_R = predict_err_abs_R
            predict_err_masked_R = predict_err_R * mask
            # Mean Absolute Error
            predict_MAE_R = np.sum(predict_err_abs_R * mask) / mask_length
            depth_MAE_R = np.sum(depth_err_abs_R * mask) / mask_length
            # Mean Squared Error
            predict_MSE_R = np.sum(predict_err_sqr_R * mask) / mask_length
            depth_MSE_R = np.sum(depth_err_sqr_R * mask) / mask_length
            # Root Mean Square Error
            predict_RMSE_R = np.sqrt(predict_MSE_R)
            depth_RMSE_R = np.sqrt(depth_MSE_R)
            #################################################################

        predict_depth = decode_img[:, :, 0].copy()
        if is_pred_smooth:
            predict_depth = common_tools.gaussian_filter(predict_depth, 2, 0.002)
        depth_gt_masked = depth_gt * mask
        gt_diff = (depth_gt - depth_gap) * mask
        predict_masked = predict_depth * mask
        # scale
        predict_depth /= difference_scaling
        predict_masked /= difference_scaling
        # predict normalization
        if is_predict_norm:
            mean_gt = np.sum(gt_diff) / mask_length
            mean_predict = np.sum(predict_masked) / mask_length
            gt_diff -= mean_gt
            predict_depth -= mean_predict
            out_diff = predict_depth.copy() # save diff
            sd_gt = np.sqrt(np.sum(np.square((gt_diff)*mask)) / mask_length)
            sd_predict = np.sqrt(np.sum(np.square((predict_depth)*mask)) / mask_length)
            predict_depth *= sd_gt / sd_predict
            predict_depth += mean_gt
            predict_masked = predict_depth * mask

        # difference learn
        predict_depth += depth_gap
        predict_masked += depth_gap * mask

        # ajust bias, calc error
        if is_pred_ajust:
            mean_gt_diff = np.sum(gt_diff) / mask_length
            mean_out_dif = np.sum(out_diff * mask) / mask_length
            bias_out_dif = mean_out_dif - mean_gt_diff
            out_diff_ajusted = out_diff - bias_out_dif

        # error
        depth_err_abs = np.abs(depth_gt - depth_gap)
        depth_err_sqr = np.square(depth_gt - depth_gap)
        if is_pred_ajust:
            predict_err_abs = np.abs(gt_diff - out_diff_ajusted)
            predict_err_sqr = np.square(gt_diff - out_diff_ajusted)
        else:
            predict_err_abs = np.abs(depth_gt - predict_depth)
            predict_err_sqr = np.square(depth_gt - predict_depth)

        # error image
        depth_err = depth_err_abs
        predict_err = predict_err_abs
        predict_err_masked = predict_err * mask
        # Mean Absolute Error
        predict_MAE = np.sum(predict_err_abs * mask) / mask_length
        depth_MAE = np.sum(depth_err_abs * mask) / mask_length
        # Mean Squared Error
        predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
        depth_MSE = np.sum(depth_err_sqr * mask) / mask_length
        # Root Mean Square Error
        predict_RMSE = np.sqrt(predict_MSE)
        depth_RMSE = np.sqrt(depth_MSE)

        if is_pred_pix_reverse:
            predict_err_abs = np.where(predict_err_abs < predict_err_abs_R, predict_err_abs, predict_err_abs_R)
            predict_err_sqr = np.where(predict_err_sqr < predict_err_sqr_R, predict_err_sqr, predict_err_sqr_R)
            predict_err = predict_err_abs
            predict_err_masked = predict_err * mask
            predict_MAE = np.sum(predict_err_abs * mask) / mask_length
            predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
            predict_RMSE = np.sqrt(predict_MSE)
        elif is_pred_reverse:
            if predict_RMSE > predict_RMSE_R:
                depth_err = depth_err_R
                predict_err = predict_err_R
                predict_err_masked = predict_err_masked_R
                predict_MAE = predict_MAE_R
                depth_MAE = depth_MAE_R
                predict_MSE = predict_MSE_R
                depth_MSE = depth_MSE_R
                predict_RMSE = predict_RMSE_R
                depth_RMSE = depth_RMSE_R
                out_diff = out_diff_R

        # output difference
        if is_save_diff:
            net_out_dir = predict_dir + '/net_output/'
            os.makedirs(net_out_dir, exist_ok=True)
            if test_idx in save_img_range:
                np.save(net_out_dir + '{:05d}.npy'.format(test_idx), out_diff)
                out_diff_depth = out_diff + 1
                xyz_out_diff = depth_tools.convert_depth_to_coords(out_diff_depth, cam_params)
                depth_tools.dump_ply(net_out_dir + '{:05d}.ply'.format(test_idx), xyz_out_diff.reshape(-1, 3).tolist())


        err_strings += str(test_idx)
        if test_idx in test_range:
        # if test_idx not in train_range:
            err_strings += ',test,'
        else:
            err_strings += ',train,'
        for string in [depth_MAE, predict_MAE,depth_RMSE, predict_RMSE]:
            err_strings += str(string) + ','
        err_strings.rstrip(',')
        err_strings = err_strings[:-1] + '\n'

        predict_loss = predict_MAE
        depth_loss = depth_MAE

        # save predicted depth
        if test_idx in save_img_range:
            predict_bmp = depth_tools.pack_float_to_bmp_bgra(predict_masked)
            cv2.imwrite(predict_dir + '/predict-{:03d}.bmp'.format(test_idx),
                        predict_bmp)

        # save ply
        if is_save_ply:
            if test_idx in save_ply_range:
                if is_masked_ply:
                    xyz_predict_masked = depth_tools.convert_depth_to_coords(predict_masked, cam_params)
                    depth_tools.dump_ply(predict_dir + '/predict_masked-%03d.ply'%test_idx, xyz_predict_masked.reshape(-1, 3).tolist())
                else:
                    xyz_predict = depth_tools.convert_depth_to_coords(predict_depth, cam_params)
                    depth_tools.dump_ply(predict_dir + '/predict-%03d.ply'%test_idx, xyz_predict.reshape(-1, 3).tolist())
                
        # save fig
        # if test_idx in test_range:
        if test_idx in save_img_range:
            # layout
            fig = plt.figure(figsize=(7, 4))
            gs_master = GridSpec(nrows=2,
                                ncols=2,
                                height_ratios=[1, 1],
                                width_ratios=[3, 0.1])
            gs_1 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=3,
                                        subplot_spec=gs_master[0, 0],
                                        wspace=0.05,
                                        hspace=0)
            gs_2 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=3,
                                        subplot_spec=gs_master[1, 0],
                                        wspace=0.05,
                                        hspace=0)
            gs_3 = GridSpecFromSubplotSpec(nrows=2,
                                        ncols=1,
                                        subplot_spec=gs_master[0:1, 1])

            ax_enh0 = fig.add_subplot(gs_1[0, 0])
            ax_enh1 = fig.add_subplot(gs_1[0, 1])
            ax_enh2 = fig.add_subplot(gs_1[0, 2])

            ax_misc0 = fig.add_subplot(gs_2[0, 0])

            ax_err_gap = fig.add_subplot(gs_2[0, 1])
            ax_err_pred = fig.add_subplot(gs_2[0, 2])

            ax_cb0 = fig.add_subplot(gs_3[0, 0])
            ax_cb1 = fig.add_subplot(gs_3[1, 0])

            for ax in [
                    ax_enh0, ax_enh1, ax_enh2,
                    ax_misc0, ax_err_gap, ax_err_pred
            ]:
                ax.axis('off')

            # close up
            mean = np.sum(depth_gt_masked) / mask_length
            vmin_s, vmax_s = mean - vm_range, mean + vm_range

            ax_enh0.imshow(depth_gt_masked, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh1.imshow(depth_gap * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh2.imshow(predict_masked, cmap='jet', vmin=vmin_s, vmax=vmax_s)

            # misc
            # ax_misc0.imshow(shading_bgr[:, :, ::-1])
            ax_misc0.imshow(np.dstack([shading_gray, shading_gray, shading_gray]))

            # error
            vmin_e, vmax_e = 0, vm_e_range
            ax_err_gap.imshow(depth_err * mask, cmap='jet', vmin=vmin_e, vmax=vmax_e)
            ax_err_pred.imshow(predict_err_masked, cmap='jet', vmin=vmin_e, vmax=vmax_e)

            # title
            # ax_enh0.set_title('Groud Truth')
            # ax_enh1.set_title('Input Depth')
            # ax_enh2.set_title('Predict')
            # ax_err_gap.set_title('')
            # ax_err_pred.set_title('')

            # colorbar
            plt.tight_layout()
            fig.savefig(io.BytesIO())
            cb_offset = -0.05

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_s, vmax=vmax_s),
                                        cmap='jet'),
                        cax=ax_cb0)
            im_pos, cb_pos = ax_enh2.get_position(), ax_cb1.get_position()
            ax_cb0.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
                                        cmap='jet'),
                        cax=ax_cb1)
            im_pos, cb_pos = ax_err_pred.get_position(), ax_cb1.get_position()
            ax_cb1.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])

            plt.savefig(predict_dir + '/result-{:03d}.png'.format(test_idx), dpi=300)
            plt.close()

    with open(predict_dir + '/error_compare.txt', mode='w') as f:
        f.write(err_strings)

    compare_error.compare_error(predict_dir + '/')
    compare_error.compare_error(predict_dir + '/', error='MAE')

if __name__ == "__main__":
    main()
