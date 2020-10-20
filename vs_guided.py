import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import io

import depth_tools

is_vs = True
# is_vs = False

batch_shape = (255, 255)
norm_patch_size = 3

DIR = '../data/vs_guided/'
GT = DIR + 'gt/'
REC = DIR + 'rec/'
SHD = DIR + 'shading/'
if is_vs:
    PRED = DIR + 'guided/'
else:
    PRED = DIR + 'proposed/'

depth_threshold = 0.8
difference_threshold = 0.005

vmin, vmax = (0.8, 1.4)
vm_range = 0.02
vm_e_range = 0.005
vm_e_range = 0.001
# vm_e_range = difference_threshold

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

list_MAE_rec =[]
list_MAE_pred = []
list_RMSE_rec = []
list_RMSE_pred = []

err_strings = 'index,MAE depth,MAE predict,RMSE depth,RMSE predict\n'

# for idx in range(16):
for idx in range(19):
    gt = np.load(GT + 'gt{:03d}.npy'.format(idx))
    rec = np.load(REC + 'depth{:03d}.npy'.format(idx))
    if is_vs:
        pred = np.load(PRED + 'predict-{:03d}.npy'.format(idx))
    else:
        pred = np.load(PRED + 'proposed{:03d}.npy'.format(idx))
    shading_gray = cv2.imread(SHD + 'shading{:03d}.png'.format(idx), 0)

    is_gt_available = (gt > depth_threshold) * 1.0
    is_rec_available = (rec > depth_threshold) * 1.0
    is_pred_available = (pred > depth_threshold) * 1.0
    is_depth_available = is_gt_available * is_rec_available * is_pred_available

    is_rec_close = (np.abs(gt - rec) < difference_threshold) * 1.0
    is_pred_close = (np.abs(gt - pred) < difference_threshold) * 1.0
    mask = is_rec_close * is_pred_close * is_depth_available

    len_mask = np.sum(mask)
    # print(len_mask)

    # xyz_gt = depth_tools.convert_depth_to_coords(gt * mask, cam_params)
    # xyz_rec = depth_tools.convert_depth_to_coords(rec * mask, cam_params)
    # xyz_pred = depth_tools.convert_depth_to_coords(pred * mask, cam_params)
    
    # depth_tools.dump_ply(DIR + 'ply_gt/gt{:03d}.ply'.format(i), xyz_gt.reshape(-1, 3).tolist())
    # depth_tools.dump_ply(DIR + 'ply_rec/rec{:03d}.ply'.format(i), xyz_rec.reshape(-1, 3).tolist())
    # if is_vs:
    #     depth_tools.dump_ply(DIR + 'ply_pred/pred{:03d}.ply'.format(i), xyz_pred.reshape(-1, 3).tolist())
    # else:
    #     depth_tools.dump_ply(DIR + 'ply_proposed/proposed{:03d}.ply'.format(i), xyz_pred.reshape(-1, 3).tolist())
    

    mask_length = len_mask

    gt_masked = gt * mask
    gt_diff = (gt - rec) * mask
    predict_masked = pred * mask

    p = norm_patch_size
    if is_vs:
        predict_diff = pred - rec
        predict_diff_masked = predict_diff * mask
        new_pred_diff = np.zeros_like(predict_diff)
        for i in range(p + 1, batch_shape[0] - p - 1):
            for j in range(p + 1, batch_shape[1] - p - 1):
                if not mask[i, j]:
                    new_pred_diff[i, j] = 0
                    continue
                local_mask = mask[i-p:i+p, j-p:j+p]
                local_gt_diff = gt_diff[i-p:i+p, j-p:j+p]
                local_pred = predict_diff_masked[i-p:i+p, j-p:j+p]
                local_mask_len = np.sum(local_mask)
                if local_mask_len < 10:
                    new_pred_diff[i, j] = 0
                    mask[i, j] = False
                    continue
                local_mean_gt = np.sum(local_gt_diff) / local_mask_len
                local_mean_pred = np.sum(local_pred) / local_mask_len
                local_sd_gt = np.sqrt(np.sum(np.square(local_gt_diff)) / local_mask_len)
                local_sd_pred = np.sqrt(np.sum(np.square(local_pred)) / local_mask_len)
                new_pred_diff[i, j] = (predict_diff[i, j] - local_mean_pred) * (local_sd_gt / local_sd_pred) + local_mean_gt
        predict_diff = new_pred_diff

    predict_depth = (new_pred_diff + rec) * mask

    depth_gt = gt.copy()
    depth_gap = rec.copy()
    # predict_depth = pred.copy() - rec
    # predict_depth = pred.copy()
    # predict_masked = predict_depth * mask

    # mean_gt = np.sum(gt_diff) / mask_length
    # mean_predict = np.sum(predict_masked) / mask_length
    # gt_diff -= mean_gt
    # predict_depth -= mean_predict
    # predict_depth *= -1.0 # reverse
    # out_diff_R = predict_depth.copy() # save diff
    # sd_gt = np.sqrt(np.sum(np.square((gt_diff)*mask)) / mask_length)
    # sd_predict = np.sqrt(np.sum(np.square((predict_depth)*mask)) / mask_length)
    # predict_depth *= sd_gt / sd_predict
    # predict_depth += mean_gt
    # predict_masked = predict_depth * mask

    # predict_depth += depth_gap
    # predict_masked += depth_gap * mask

    # depth_err_abs_R = np.abs(depth_gt - depth_gap)
    # depth_err_sqr_R = np.square(depth_gt - depth_gap)
    # predict_err_abs_R = np.abs(depth_gt - predict_depth)
    # predict_err_sqr_R = np.square(depth_gt - predict_depth)

    # error image
    # depth_err_R = depth_err_abs_R
    # predict_err_R = predict_err_abs_R
    # predict_err_masked_R = predict_err_R * mask
    # # Mean Absolute Error
    # predict_MAE_R = np.sum(predict_err_abs_R * mask) / mask_length
    # depth_MAE_R = np.sum(depth_err_abs_R * mask) / mask_length
    # # Mean Squared Error
    # predict_MSE_R = np.sum(predict_err_sqr_R * mask) / mask_length
    # depth_MSE_R = np.sum(depth_err_sqr_R * mask) / mask_length
    # # Root Mean Square Error
    # predict_RMSE_R = np.sqrt(predict_MSE_R)
    # depth_RMSE_R = np.sqrt(depth_MSE_R)
    # #################################################################
    # predict_depth = pred.copy() - rec

    # depth_gt_masked = depth_gt * mask
    # gt_diff = (depth_gt - depth_gap) * mask
    # predict_masked = predict_depth * mask

    # mean_gt = np.sum(gt_diff) / mask_length
    # mean_predict = np.sum(predict_masked) / mask_length
    # gt_diff -= mean_gt
    # predict_depth -= mean_predict
    # out_diff = predict_depth.copy() # save diff
    # sd_gt = np.sqrt(np.sum(np.square((gt_diff)*mask)) / mask_length)
    # sd_predict = np.sqrt(np.sum(np.square((predict_depth)*mask)) / mask_length)
    # predict_depth *= sd_gt / sd_predict
    # predict_depth += mean_gt
    # predict_masked = predict_depth * mask

    # predict_depth += depth_gap
    # predict_masked += depth_gap * mask

    depth_err_abs = np.abs(depth_gt - depth_gap)
    depth_err_sqr = np.square(depth_gt - depth_gap)

    predict_err_abs = np.abs(depth_gt - predict_depth)
    predict_err_sqr = np.square(depth_gt - predict_depth)

    # error image
    # depth_err = depth_err_abs
    # predict_err = predict_err_abs
    # predict_err_masked = predict_err * mask
    depth_err = gt_diff
    predict_err = gt - predict_depth
    predict_err_masked = np.zeros_like(predict_err)
    predict_err_masked[p+1:batch_shape[0] - p-1, p+1:batch_shape[1] - p-1] = predict_err[p+1:batch_shape[0] - p-1, p+1:batch_shape[1] - p-1]
    predict_err_masked *= mask
    # Mean Absolute Error
    predict_MAE = np.sum(predict_err_abs * mask) / mask_length
    depth_MAE = np.sum(depth_err_abs * mask) / mask_length
    # Mean Squared Error
    predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
    depth_MSE = np.sum(depth_err_sqr * mask) / mask_length
    # Root Mean Square Error
    predict_RMSE = np.sqrt(predict_MSE)
    depth_RMSE = np.sqrt(depth_MSE)

    # predict_err_abs = np.where(predict_err_abs < predict_err_abs_R, predict_err_abs, predict_err_abs_R)
    # predict_err_sqr = np.where(predict_err_sqr < predict_err_sqr_R, predict_err_sqr, predict_err_sqr_R)

    # predict_err = predict_err_abs
    # predict_err_masked = predict_err * mask
    # predict_MAE = np.sum(predict_err_abs * mask) / mask_length
    # predict_MSE = np.sum(predict_err_sqr * mask) / mask_length
    # predict_RMSE = np.sqrt(predict_MSE)



    # err_rec_abs = np.abs(gt - rec) * mask
    # err_pred_abs = np.abs(gt - pred) * mask
    # err_rec_sqr = np.square(gt - rec) * mask
    # err_pred_sqr = np.square(gt - pred) * mask

    # MAE_rec = np.sum(err_rec_abs) / len_mask
    # MAE_pred = np.sum(err_pred_abs) / len_mask

    # MSE_rec = np.sum(err_rec_sqr) / len_mask
    # MSE_pred = np.sum(err_pred_sqr) / len_mask

    # RMSE_rec = np.sqrt(MSE_rec)
    # RMSE_pred = np.sqrt(MSE_pred)

    # list_MAE_rec.append(MAE_rec)
    # list_MAE_pred.append(MAE_pred)
    # list_RMSE_rec.append(RMSE_rec)
    # list_RMSE_pred.append(RMSE_pred)

    list_MAE_rec.append(depth_MAE)
    list_MAE_pred.append(predict_MAE)
    list_RMSE_rec.append(depth_RMSE)
    list_RMSE_pred.append(predict_RMSE)

    err_strings += str(idx) + ','
    for string in [depth_MAE, predict_MAE, depth_RMSE, predict_RMSE]:
        err_strings += str(string) + ','
    err_strings.rstrip(',')
    err_strings = err_strings[:-1] + '\n'



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
    mean = np.sum(gt * mask) / len_mask
    vmin_s, vmax_s = mean - vm_range, mean + vm_range

    ax_enh0.imshow(gt * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)
    ax_enh1.imshow(rec * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)
    ax_enh2.imshow(predict_depth * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)

    # misc
    # ax_misc0.imshow(shading_bgr[:, :, ::-1])
    ax_misc0.imshow(np.dstack([shading_gray, shading_gray, shading_gray]))

    # error
    # vmin_e, vmax_e = 0, vm_e_range
    # ax_err_gap.imshow(depth_err * mask, cmap='jet', vmin=vmin_e, vmax=vmax_e)
    # ax_err_pred.imshow(predict_err_masked, cmap='jet', vmin=vmin_e, vmax=vmax_e)
    vmin_e, vmax_e = -1 * vm_e_range, vm_e_range
    ax_err_gap.imshow(depth_err * mask, cmap='coolwarm', vmin=vmin_e, vmax=vmax_e)
    ax_err_pred.imshow(predict_err_masked, cmap='coolwarm', vmin=vmin_e, vmax=vmax_e)

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

    # plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
    #                             cmap='jet'),
    #             cax=ax_cb1)
    plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
                                cmap='coolwarm'),
                cax=ax_cb1)
    im_pos, cb_pos = ax_err_pred.get_position(), ax_cb1.get_position()
    ax_cb1.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])

    if is_vs:
        plt.savefig(DIR + 'output_vs_guided/result-{:03d}.png'.format(idx), dpi=300)
    else:
        plt.savefig(DIR + 'output_proposed/result-{:03d}.png'.format(idx), dpi=300)
    plt.close()



err_strings += 'Avg,'
for string in [np.mean(list_MAE_rec), np.mean(list_MAE_pred), 
                np.mean(list_RMSE_rec), np.mean(list_RMSE_pred)]:
    err_strings += str(string) + ','
err_strings.rstrip(',')
err_strings = err_strings[:-1] + '\n'

if is_vs:
    with open(DIR + 'output_vs_guided/vs_guided_err.txt', mode='w') as f:
        f.write(err_strings)
else:
    with open(DIR + 'output_proposed/proposed_err.txt', mode='w') as f:
        f.write(err_strings)