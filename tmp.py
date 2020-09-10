import sys
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock
import time
from tqdm import tqdm
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


for data_idx in tqdm(range(68)):

    src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
    # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(data_idx)
    src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
    # src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)
    src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)


    depth_img_gap = cv2.imread(src_depth_gap, -1)
    new_rec = np.zeros_like(depth_img_gap)
    crop = 700
    new_rec[:1200, crop:1200, :] = depth_img_gap[:1200, crop:1200, :]
    new_rec_file = src_dir + '/rec/depth{:03d}.bmp'.format(data_idx)
    cv2.imwrite(new_rec_file, new_rec)


    depth_gap = depth_tools.unpack_bmp_bgra_to_float(new_rec)

    depth_img_gt = cv2.imread(src_depth_gt, -1)
    depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)

    # difference
    difference = depth_gt - depth_gap
    # mask
    is_gap_available = depth_gap > depth_threshold
    is_depth_close = np.logical_and(
            np.abs(difference) < difference_threshold,
            is_gap_available)
    mask = is_depth_close.astype(np.float32)
    length = np.sum(mask)

    dif_masked = difference * mask

    vmin_e, vmax_e = 0, vm_e_range
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(dif_masked, cmap='jet', vmin=vmin_e, vmax=vmax_e)
    fig.savefig(src_dir + '/depth_dif/{:03d}.png'.format(data_idx))
    plt.close()