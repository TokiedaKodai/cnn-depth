import sys
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock
import time
from tqdm import tqdm
import shutil

import depth_tools

src_dir = '../data/board'

src_rec_dir = src_dir + '/rec'
src_rec_dir = src_dir + '/rec_ajusted'
src_frame_dir = src_dir + '/frame'
src_gt_dir = src_dir + '/gt'
src_shading_dir = src_dir + '/shading'

depth_threshold = 0.2
difference_threshold = 0.01
vm_e_range = 0.002

cp_dir = '../data/render_waves_600'
src_dir = '../data/render_wave1_400'
for idx in tqdm(range(600)):
    data_idx = idx
    if idx >= 400:
        src_dir = '../data/render_wave2_1100'
        data_idx -= 400
    elif idx >= 200:
        src_dir = '../data/render_wave1-double_800'
        data_idx -= 200

    src_rec_dir = src_dir + '/rec'
    src_frame_dir = src_dir + '/proj'
    src_gt_dir = src_dir + '/gt'
    src_shading_dir = src_dir + '/shade'

    src_bgra = src_frame_dir + '/{:05d}.png'.format(data_idx)
    src_depth_gap = src_rec_dir + '/{:05d}.bmp'.format(data_idx)
    src_depth_gt = src_gt_dir + '/{:05d}.bmp'.format(data_idx)
    src_shading = src_shading_dir + '/{:05d}.png'.format(data_idx)

    cp_rec_dir = cp_dir + '/rec'
    cp_frame_dir = cp_dir + '/proj'
    cp_gt_dir = cp_dir + '/gt'
    cp_shading_dir = cp_dir + '/shade'

    cp_bgra = cp_frame_dir + '/{:05d}.png'.format(idx)
    cp_depth_gap = cp_rec_dir + '/{:05d}.bmp'.format(idx)
    cp_depth_gt = cp_gt_dir + '/{:05d}.bmp'.format(idx)
    cp_shading = cp_shading_dir + '/{:05d}.png'.format(idx)

    shutil.copyfile(src_bgra, cp_bgra)
    shutil.copyfile(src_depth_gap, cp_depth_gap)
    shutil.copyfile(src_depth_gt, cp_depth_gt)
    shutil.copyfile(src_shading, cp_shading)

    


#     depth_img_gap = cv2.imread(src_depth_gap, -1)
#     new_rec = np.zeros_like(depth_img_gap)
#     crop = 700
#     new_rec[:1200, crop:1200, :] = depth_img_gap[:1200, crop:1200, :]
#     new_rec_file = src_dir + '/rec/depth{:03d}.bmp'.format(data_idx)
#     cv2.imwrite(new_rec_file, new_rec)


#     depth_gap = depth_tools.unpack_bmp_bgra_to_float(new_rec)

#     depth_img_gt = cv2.imread(src_depth_gt, -1)
#     depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)

#     # difference
#     difference = depth_gt - depth_gap
#     # mask
#     is_gap_available = depth_gap > depth_threshold
#     is_depth_close = np.logical_and(
#             np.abs(difference) < difference_threshold,
#             is_gap_available)
#     mask = is_depth_close.astype(np.float32)
#     length = np.sum(mask)

#     dif_masked = difference * mask

#     vmin_e, vmax_e = 0, vm_e_range
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111)
#     ax.axis("off")
#     ax.imshow(dif_masked, cmap='jet', vmin=vmin_e, vmax=vmax_e)
#     fig.savefig(src_dir + '/depth_dif/{:03d}.png'.format(data_idx))
#     plt.close()