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
_, out_dir, epoch_num, net_type = argv # output dir, epoch

# out_dir = '../output/output_' + out_dir
out_dir = '../output/archive/200318/output_' + out_dir

epoch_num = int(epoch_num)

# normalization
is_shading_norm = True

# parameters
depth_threshold = 0.2
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

#InputData
src_dir = '../data/input_200201'
src_dir = '../data/input_200302'
src_dir = '../data/input_200201-0312/'
src_dir = '../data/input_200318'

# save predict depth PLY file
is_save_ply = True
# is_save_ply = False

is_masked_ply = True

# predict normalization
# is_predict_norm = True 
is_predict_norm = False
# is_predict_norm = int(is_predict_norm)

# select from val loss
is_select_val = True
is_select_val = False

# if is_predict_norm:
#     if is_select_val:
#         predict_dir = out_dir + '/predict_{}_norm'.format(epoch_num)
#     else:
#         predict_dir = out_dir + '/predict_{}_norm_loss'.format(epoch_num)
# else:
#     if is_select_val:
#         predict_dir = out_dir + '/predict_{}'.format(epoch_num)
#         # predict_dir = out_dir + '/predict_{}_fake'.format(epoch_num)
#     else:
#         predict_dir = out_dir + '/predict_{}_loss'.format(epoch_num)

predict_dir = out_dir + '/predict_{}'.format(epoch_num)
predict_dir = out_dir + '/predict_{}_test'.format(epoch_num)

# data_num = 6
# data_num = 40
# data_num = 60
data_num = 56
data_num = 68
# if epoch_num == 1000:
    # is_save_ply = True
    # data_num = 60

data_idx_range = list(range(data_num))
# data_idx_range.extend(list(range(48, 52)))
# data_idx_range = range(40, 60)
# data_idx_range = list()
# for i in range(5):
#     data_idx_range.extend([0+8*i, 1+8*i, 3+8*i, 6+8*i])

'''
Test Data
ori 110cm : 16 - 23
small 100cm : 44 - 47
mid 110cm : 56 - 59
'''
# test data
# test_range = list(range(40))
test_range = list(range(16, 24))
test_range.extend(list(range(44, 48)))
# test_range.extend(list(range(48, 56)))
test_range.extend(list(range(56, 60)))

# test_range.extend(list(range(48, 52)))
# test_range = list()
# for i in range(5):
#     test_range.extend([3+8*i, 6+8*i])

# train data
'''no-fake data'''
# train_range = list(range(16))
# train_range.extend(list(range(24, 40)))
# train_range.extend(list(range(40, 48)))
# train_range.extend(list(range(56, 58)))
'''fake data'''
# train_range.extend(list(range(40, 48)))
# train_range.extend(list(range(52, 60)))
'''no-rotate data'''
# train_range = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33]

# train_range = list()
# for i in range(5):
#     train_range.extend([0+8*i, 1+8*i, 3+8*i, 6+8*i])

# save ply range
save_ply_range = test_range

vmin, vmax = (0.8, 1.4)
vm_range = 0.05
# vm_e_range = 0.01
vm_e_range = 0.005

batch_shape = (1200, 1200)
# batch_shape = (600, 600)
batch_tl = (0, 0)  # top, left

# input_200201
cam_params = {
    'focal_length': 0.037306625,
    'pix_x': 1.25e-05,
    'pix_y': 1.2360472397638345e-05,
    'center_x': 801.557,
    'center_y': 555.618
}

def main():
    # src_rec_dir = src_dir + '/rec'
    src_rec_dir = src_dir + '/rec_ajusted'
    src_frame_dir = src_dir + '/frame'
    src_gt_dir = src_dir + '/gt'
    src_shading_dir = src_dir + '/shading'

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
    for test_idx in tqdm(data_idx_range):
        src_bgra = src_frame_dir + '/frame{:03d}.png'.format(test_idx)
        # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(test_idx)
        src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(test_idx)
        src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(test_idx)
        # src_shading = src_shading_dir + '/shading{:03d}.png'.format(test_idx)
        src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(test_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        depth_gt = depth_gt[:, :1200]  # clipping

        shading_bgr = cv2.imread(src_shading, -1)
        # shading = cv2.imread(src_shading, 0) # GrayScale
        shading = np.zeros_like(shading_bgr)
        shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]

        if is_shading_norm:
            shading = shading / np.max(shading)
        else:
            shading = shading / 255.

        img_shape = bgr.shape[:2]

        # normalization (may not be needed)
        # norm_factor = depth_gap.max()
        # depth_gap /= norm_factor
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

        mask_gt = is_gt_available * 1.0

        # cv2.imwrite(predict_dir + '/mask-{:05d}.png'.format(test_idx),
        #             (mask * 255).astype(np.uint8))

        predict_depth = decode_img[:, :, 0].copy()

        depth_gt_masked = depth_gt * mask
        depth_gt_zero = (depth_gt - depth_gap) * mask
        predict_masked = predict_depth * mask

        # scale
        predict_depth /= difference_scaling
        predict_masked /= difference_scaling

        # predict normalization
        if is_predict_norm:
            mean_gt = np.mean(depth_gt_zero)
            mean_predict = np.mean(predict_masked)
            depth_gt_zero -= mean_gt
            predict_depth -= mean_predict
            predict_masked -= mean_predict
            sd_gt = np.sqrt(np.var(depth_gt_zero))
            sd_predict = np.sqrt(np.var(predict_masked))
            predict_depth *= sd_gt / sd_predict
            predict_masked *= sd_gt / sd_predict
            depth_gt = (depth_gt_zero + depth_gap) * mask

        # difference learn
        predict_depth += depth_gap
        predict_masked += depth_gap * mask

        # predict_depth = common_tools.gaussian_filter(predict_depth, 4)
        # predict_masked = predict_depth * mask

        # error
        depth_err_abs = np.abs(depth_gt - depth_gap)
        predict_err_abs = np.abs(depth_gt - predict_depth)
        depth_err_sqr = np.square(depth_gt - depth_gap)
        predict_err_sqr = np.square(depth_gt - predict_depth)
        # error image
        depth_err = depth_err_abs
        predict_err = predict_err_abs
        predict_err_masked = predict_err * mask

        mask_length = np.sum(mask)

        # Mean Absolute Error
        predict_MAE = np.sum(predict_err_abs * mask) / mask_length
        depth_MAE = np.sum(depth_err_abs * mask) / mask_length

        # Mean Squared Error
        predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
        depth_MSE = np.sum(depth_err_sqr * mask) / mask_length

        # Root Mean Square Error
        predict_RMSE = np.sqrt(predict_MSE)
        depth_RMSE = np.sqrt(depth_MSE)

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
        if test_idx in save_ply_range:
            predict_bmp = depth_tools.pack_float_to_bmp_bgra(predict_depth)
            cv2.imwrite(predict_dir + '/predict_depth-{:03d}.bmp'.format(test_idx),
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
        if test_idx in test_range:
        # if test_idx not in train_range:
            # layout
            fig = plt.figure(figsize=(8, 6))
            gs_master = GridSpec(nrows=3,
                                ncols=2,
                                height_ratios=[1, 1, 1],
                                width_ratios=[4, 0.1])
            gs_1 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=4,
                                        subplot_spec=gs_master[0, 0],
                                        wspace=0.05,
                                        hspace=0)
            gs_2 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=4,
                                        subplot_spec=gs_master[1, 0],
                                        wspace=0.05,
                                        hspace=0)

            gs_3 = GridSpecFromSubplotSpec(nrows=1,
                                        ncols=4,
                                        subplot_spec=gs_master[2, 0],
                                        wspace=0.05,
                                        hspace=0)

            gs_4 = GridSpecFromSubplotSpec(nrows=3,
                                        ncols=1,
                                        subplot_spec=gs_master[0:2, 1])

            ax_reg0 = fig.add_subplot(gs_1[0, 0])
            ax_reg1 = fig.add_subplot(gs_1[0, 1])
            ax_reg2 = fig.add_subplot(gs_1[0, 2])
            ax_reg3 = fig.add_subplot(gs_1[0, 3])

            ax_enh0 = fig.add_subplot(gs_2[0, 0])
            ax_enh1 = fig.add_subplot(gs_2[0, 1])
            ax_enh2 = fig.add_subplot(gs_2[0, 2])
            ax_enh3 = fig.add_subplot(gs_2[0, 3])

            ax_misc0 = fig.add_subplot(gs_3[0, 0])

            ax_cb0 = fig.add_subplot(gs_4[0, 0])
            ax_cb1 = fig.add_subplot(gs_4[1, 0])

            # rmse
            ax_err_gap = fig.add_subplot(gs_3[0, 1])
            ax_err = fig.add_subplot(gs_3[0, 2])
            ax_err_masked = fig.add_subplot(gs_3[0, 3])
            ax_cb2 = fig.add_subplot(gs_4[2, 0])

            for ax in [
                    ax_reg0, ax_reg1, ax_reg2, ax_reg3, ax_enh0, ax_enh1, ax_enh2,
                    ax_enh3, ax_misc0, ax_err_gap, ax_err, ax_err_masked
            ]:
                ax.axis('off')

            ax_reg0.imshow(depth_gt, vmin=vmin, vmax=vmax)
            ax_reg1.imshow(depth_gap, vmin=vmin, vmax=vmax)
            ax_reg2.imshow(predict_depth, vmin=vmin, vmax=vmax)
            ax_reg3.imshow(predict_masked, vmin=vmin, vmax=vmax)

            # close up
            # mean = np.median(depth_gt)
            mean = np.sum(depth_gt_masked) / mask_length
            # vmin_s, vmax_s = mean - 0.05, mean + 0.05
            vmin_s, vmax_s = mean - vm_range, mean + vm_range

            ax_enh0.imshow(depth_gt, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh1.imshow(depth_gap, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh2.imshow(predict_depth, cmap='jet', vmin=vmin_s, vmax=vmax_s)
            ax_enh3.imshow(predict_masked, cmap='jet', vmin=vmin_s, vmax=vmax_s)

            # misc
            ax_misc0.imshow(shading_bgr[:, :, ::-1])
            # ax_misc0.imshow(shading[:, :, ::-1])
            # ax_misc1.imshow(train_segment, cmap='rainbow', vmin=0, vmax=2)

            # error
            # vmin_e, vmax_e = 0, 0.05
            # vmin_e, vmax_e = 0, 0.02
            # vmin_e, vmax_e = 0, 0.01
            vmin_e, vmax_e = 0, vm_e_range
            ax_err_gap.imshow(depth_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)
            ax_err.imshow(predict_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)
            ax_err_masked.imshow(predict_err_masked, cmap='jet', vmin=vmin_e, vmax=vmax_e)

            # title
            ax_reg0.set_title('GT')
            ax_reg1.set_title('Depth')
            ax_reg2.set_title('Predict')
            ax_reg3.set_title('Masked predict')

            if test_idx in test_range:
            # if test_idx not in train_range:
                ax_enh0.set_title('Test data')
            else:
                ax_enh0.set_title('Train data')
            ax_enh1.set_title('Train epoch:{}'.format(epoch_num))
            ax_enh2.set_title('Model epoch:{}'.format(idx_min_loss))
            ax_enh3.set_title('Train loss:{:.6f}'.format(min_loss))

            ax_err_gap.set_title('Depth error:{:.6f}'.format(depth_loss))
            ax_err_masked.set_title('Predict error:{:.6f}'.format(predict_loss))

            # colorbar
            plt.tight_layout()
            fig.savefig(io.BytesIO())
            cb_offset = -0.05

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax)),
                        cax=ax_cb0)
            im_pos, cb_pos = ax_reg3.get_position(), ax_cb0.get_position()
            ax_cb0.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_s, vmax=vmax_s),
                                        cmap='jet'),
                        cax=ax_cb1)
            im_pos, cb_pos = ax_enh3.get_position(), ax_cb1.get_position()
            ax_cb1.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])

            plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
                                        cmap='jet'),
                        cax=ax_cb2)
            im_pos, cb_pos = ax_err.get_position(), ax_cb2.get_position()
            ax_cb2.set_position([
                cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
                im_pos.y1 - im_pos.y0
            ])

            plt.savefig(predict_dir + '/compare-{:03d}.png'.format(test_idx), dpi=300)
            plt.close()

    with open(predict_dir + '/error_compare.txt', mode='w') as f:
        f.write(err_strings)

    compare_error.compare_error(predict_dir + '/')

if __name__ == "__main__":
    main()
