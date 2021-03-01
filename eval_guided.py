import numpy as np
import cv2
import io
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable

import depth_tools

dir_data = '../data/'
dir_pred = '../output/guided/'

data_type = 'board'
idx_range = range(40, 56)

data_type = 'real'
idx_range = range(19)

dir_data += data_type
dir_pred += data_type
dir_out = dir_pred + '/result'
dir_guided = dir_pred + '/guided'


is_norm_guided = True
# is_norm_guided = False

depth_thre = 0.2
diff_thre = 0.005

norm_patch_size = 24
patch_rate = 50

res = 512
resp = 256

vmin, vmax = (0.8, 1.4)
# vm_range = 0.03
vm_range = 0.01
vm_e_range = 0.002

cam_params = {
    'focal_length': 0.037009,
    'pix_x': 1.25e-05,
    'pix_y': 1.2381443057539635e-05,
    'center_x': 790.902,
    'center_y': 600.635
}

result_str = 'idx,Gapreconst,Pred,Guided\n'

dir_ply = dir_out + '/ply/'
os.makedirs(dir_out, exist_ok=True)
os.makedirs(dir_ply, exist_ok=True)

idxg = 0
for idx in tqdm(idx_range):
    # pred = np.load(dir_data + '/proposed_patch/proposed{:03d}.npy'.format(idx))
    # guided = np.load(dir_guided + '/predict-{:03d}.npy'.format(idx))
    # gt = np.load(dir_data + '/gt_patch/gt{:03d}.npy'.format(idx))
    # rec = np.load(dir_data + '/rec_patch/depth{:03d}.npy'.format(idx))
    # shading_bgr = cv2.imread(dir_data + '/shading_patch/shading{:03d}.png'.format(idx), 1)
    # shading_gray = cv2.imread(dir_data + '/shading_patch/shading{:03d}.png'.format(idx), 0)
    pred_img = cv2.imread(dir_pred + '/clip_pred/{:05d}.bmp'.format(idx), -1)
    gt_img = cv2.imread(dir_data + '/clip_gt/{:05d}.bmp'.format(idx), -1)
    rec_img = cv2.imread(dir_data + '/clip_rec/{:05d}.bmp'.format(idx), -1)
    shading_bgr = cv2.imread(dir_data + '/clip_shade/{:05d}.png'.format(idx), 1)
    shading_gray = cv2.imread(dir_data + '/clip_shade/{:05d}.png'.format(idx), 0)

    pred = depth_tools.unpack_bmp_bgra_to_float(pred_img)
    gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)
    rec = depth_tools.unpack_bmp_bgra_to_float(rec_img)
    guided = np.zeros_like(pred)
    for i in range(2):
        for j in range(2):
            guided_patch = np.load(dir_guided + '/predict-{:03d}.npy'.format(idxg))
            guided[resp*i: resp*(i+1), resp*j: resp*(j+1)] = guided_patch
            idxg += 1

    is_pred_valid = pred > depth_thre
    is_depth_close = np.abs(gt - rec) < diff_thre
    mask = np.logical_and(is_pred_valid, is_depth_close) * 1.0
    mask_length = np.sum(mask)

    diff_gt = (gt - rec) * mask
    diff_guided = (guided - rec) * mask

    p = norm_patch_size
    new_guided = np.zeros_like(guided)
    new_mask = mask.copy()

    if is_norm_guided:
        for i in range(p + 1, res - p - 1):
            for j in range(p + 1, res - p - 1):
                if not mask[i, j]:
                    new_guided[i, j] = 0
                    continue
                local_mask = mask[i-p:i+p+1, j-p:j+p+1]
                local_gt = diff_gt[i-p:i+p+1, j-p:j+p+1]
                local_guided = diff_guided[i-p:i+p+1, j-p:j+p+1]
                local_mask_len = np.sum(local_mask)
                patch_len = (p*2 + 1) ** 2
                if local_mask_len < patch_len*patch_rate/100:
                    new_guided[i, j] = 0
                    new_mask[i, j] = False
                    continue
                local_mean_gt = np.sum(local_gt) / local_mask_len
                local_mean_guided = np.sum(local_guided) / local_mask_len
                local_sd_gt = np.sqrt(np.sum(np.square(local_gt - local_mean_gt)) / local_mask_len)
                local_sd_guided = np.sqrt(np.sum(np.square(local_guided - local_mean_guided)) / local_mask_len)
                new_guided[i, j] = (diff_guided[i, j] - local_mean_guided) * (local_sd_gt / local_sd_guided) + local_mean_gt

        mask = new_mask
        mask[:p + 1, :] = False
        mask[res - p - 1:, :] = False
        mask[:, :p + 1] = False
        mask[:, res - p - 1:] = False
        mask *= 1.0
        mask_length = np.sum(mask)

        guided = (new_guided + rec) * mask

    gt = gt[p + 1:res - p - 1, p + 1:res - p - 1]
    rec = rec[p + 1:res - p - 1, p + 1:res - p - 1]
    pred = pred[p + 1:res - p - 1, p + 1:res - p - 1]
    guided = guided[p + 1:res - p - 1, p + 1:res - p - 1]
    mask = mask[p + 1:res - p - 1, p + 1:res - p - 1]
    shading_gray = shading_gray[p + 1:res - p - 1, p + 1:res - p - 1]
    shading_bgr = shading_bgr[p + 1:res - p - 1, p + 1:res - p - 1, :]

    err_abs_rec = np.abs(gt - rec) * mask
    err_abs_pred = np.abs(gt - pred) * mask
    err_abs_guided = np.abs(gt - guided) * mask

    err_sqr_rec = np.square(gt - rec) * mask
    err_sqr_pred = np.square(gt - pred) * mask
    err_sqr_guided = np.square(gt - guided) * mask

    mse_rec = np.sum(err_sqr_rec) / mask_length
    mse_pred = np.sum(err_sqr_pred) / mask_length
    mse_guided = np.sum(err_sqr_guided) / mask_length

    rmse_rec = np.sqrt(mse_rec)
    rmse_pred = np.sqrt(mse_pred)
    rmse_guided = np.sqrt(mse_guided)

    # PLY
    xyz_gt = depth_tools.convert_depth_to_coords(gt * mask, cam_params)
    xyz_pred = depth_tools.convert_depth_to_coords(pred * mask, cam_params)
    xyz_guided = depth_tools.convert_depth_to_coords(guided * mask, cam_params)
    depth_tools.dump_ply(dir_ply + 'gt-{:03d}.ply'.format(idx), xyz_gt.reshape(-1, 3).tolist())
    depth_tools.dump_ply(dir_ply + 'pred-{:03d}.ply'.format(idx), xyz_pred.reshape(-1, 3).tolist())
    depth_tools.dump_ply(dir_ply + 'guided-{:03d}.ply'.format(idx), xyz_guided.reshape(-1, 3).tolist())

    for s in [idx, rmse_rec, rmse_pred, rmse_guided]:
        result_str += str(s) +','
    result_str = result_str[:-1] + '\n'

    # layout
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams["font.size"] = 18
    gs_master = GridSpec(nrows=2,
                        ncols=2,
                        height_ratios=[1, 1],
                        width_ratios=[3, 0.1])
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
    gs_3 = GridSpecFromSubplotSpec(nrows=2,
                                ncols=1,
                                subplot_spec=gs_master[0:1, 1])

    ax_enh0 = fig.add_subplot(gs_1[0, 0])
    ax_enh1 = fig.add_subplot(gs_1[0, 1])
    ax_enh2 = fig.add_subplot(gs_1[0, 2])
    ax_enh3 = fig.add_subplot(gs_1[0, 3])

    ax_misc0 = fig.add_subplot(gs_2[0, 0])

    ax_err_gap = fig.add_subplot(gs_2[0, 1])
    ax_err_guided = fig.add_subplot(gs_2[0, 2])
    ax_err_pred = fig.add_subplot(gs_2[0, 3])

    ax_cb0 = fig.add_subplot(gs_3[0, 0])
    ax_cb1 = fig.add_subplot(gs_3[1, 0])

    for ax in [
            ax_enh0, ax_enh1, ax_enh2, ax_enh3,
            ax_misc0, ax_err_gap, ax_err_guided, ax_err_pred
    ]:
        ax.axis('off')

    # close up
    # mean = np.sum(gt * mask) / mask_length
    # vmin_s, vmax_s = mean - vm_range, mean + vm_range
    
    # whole
    gt_in_mask = gt[np.nonzero(gt * mask)]
    vmin_s, vmax_s = np.min(gt_in_mask), np.max(gt_in_mask)

    ax_enh0.imshow(gt, cmap='jet', vmin=vmin_s, vmax=vmax_s)
    ax_enh1.imshow(rec * mask, cmap='jet', vmin=vmin_s, vmax=vmax_s)
    ax_enh2.imshow(guided, cmap='jet', vmin=vmin_s, vmax=vmax_s)
    ax_enh3.imshow(pred, cmap='jet', vmin=vmin_s, vmax=vmax_s)

    ax_enh0.set_title('Ground Truth')
    ax_enh1.set_title('Low-res')
    ax_enh2.set_title('[2]')
    ax_enh3.set_title('Ours')

    # misc
    ax_misc0.imshow(shading_bgr[:, :, ::-1])
    # ax_misc0.imshow(np.dstack([shading_gray, shading_gray, shading_gray]))

    # error
    is_scale_err_mm = True
    if is_scale_err_mm:
        scale_err = 1000
    else:
        scale_err = 1

    vmin_e, vmax_e = 0, vm_e_range * scale_err
    ax_err_gap.imshow(err_abs_rec * mask * scale_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)
    ax_err_guided.imshow(err_abs_guided * scale_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)
    ax_err_pred.imshow(err_abs_pred * scale_err, cmap='jet', vmin=vmin_e, vmax=vmax_e)

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
    ax_cb0.set_xlabel('                [m]')

    plt.colorbar(ScalarMappable(colors.Normalize(vmin=vmin_e, vmax=vmax_e),
                                cmap='jet'),
                cax=ax_cb1)
    im_pos, cb_pos = ax_err_pred.get_position(), ax_cb1.get_position()
    ax_cb1.set_position([
        cb_pos.x0 + cb_offset, im_pos.y0, cb_pos.x1 - cb_pos.x0,
        im_pos.y1 - im_pos.y0
    ])
    if is_scale_err_mm:
        ax_cb1.set_xlabel('                [mm]')
    else:
        ax_cb1.set_xlabel('                [m]')

    plt.savefig(dir_out + '/result-{:03d}.png'.format(idx), dpi=300)
    plt.savefig(dir_out + '/result-{:03d}.pdf'.format(idx), dpi=300)
    plt.savefig(dir_out + '/result-{:03d}.svg'.format(idx), dpi=300)
    plt.close()

with open(dir_out + '/error_compare.txt', mode='w') as f:
        f.write(result_str)
