import numpy as np
import cv2
import os
from tqdm import tqdm

import depth_tools as tools

dir_root_data = '../data/'

# board
dir_data = dir_root_data + 'board/'
range_idx = list(range(16))
range_idx.extend(list(range(40, 56)))
range_idx = list(range(40, 56))
file_gt = dir_data + 'gt/gt{:03d}.bmp'
file_rec = dir_data + 'rec/depth{:03d}.bmp'
file_shade = dir_data + 'shading/shading{:03d}.bmp'
file_proj = dir_data + 'frame/frame{:03d}.png'

# real
dir_data = dir_root_data + 'real/'
range_idx = range(19)
file_gt = dir_data + 'gt/{:05d}.bmp'
file_rec = dir_data + 'rec/{:05d}.bmp'
file_shade = dir_data + 'shade/{:05d}.png'
file_proj = dir_data + 'proj/{:05d}.png'

# predict
dir_out = '../output/output_wave2-pose_lumi/'
# dir_out = '../output/output_wave2-pose_lumi_FT/'
# dir_out = '../output/output_wave2-pose_lumi_TL/'
# dir_out = '../output/output_wave2-pose_lumi_TL-E/'
# dir_out = '../output/output_board-real/'

dir_pred = dir_out + 'predict_400_board-clip_norm-local-pix=24_rate=95_crop=4_vloss_min/'
dir_pred = dir_out + 'predict_400_real_norm-local-pix=24_rate=50_crop=2_vloss_min/'

file_pred = dir_pred + 'predict-{:03d}.bmp'

res_img = (1200, 1200)
res_clip = (512, 512)
center_clip = (res_clip[0] // 2, res_clip[1] // 2)
thre_depth = 0.2
thre_diff = 0.01

# save
save_gt = dir_data + 'clip_gt/'
save_rec = dir_data + 'clip_rec/'
save_shade = dir_data + 'clip_shade/'
save_proj = dir_data + 'clip_proj/'
save_mask = dir_data + 'clip_mask/'
save_pred = dir_pred + 'clip_pred/'

os.makedirs(save_gt, exist_ok=True)
os.makedirs(save_rec, exist_ok=True)
os.makedirs(save_shade, exist_ok=True)
os.makedirs(save_proj, exist_ok=True)
os.makedirs(save_mask, exist_ok=True)
os.makedirs(save_pred, exist_ok=True)

for idx in tqdm(range_idx):
    gt_img = cv2.imread(file_gt.format(idx), -1)
    rec_img = cv2.imread(file_rec.format(idx), -1)
    shade_img = cv2.imread(file_shade.format(idx), 1)
    proj_img = cv2.imread(file_proj.format(idx), 1)
    pred_img = cv2.imread(file_pred.format(idx), -1)

    gt_img = gt_img[:res_img[1], :res_img[0], :]
    rec_img = rec_img[:res_img[1], :res_img[0], :]
    shade_img = shade_img[:res_img[1], :res_img[0], :]
    proj_img = proj_img[:res_img[1], :res_img[0], :]

    gt = tools.unpack_bmp_bgra_to_float(gt_img)
    rec = tools.unpack_bmp_bgra_to_float(rec_img)
    pred = tools.unpack_bmp_bgra_to_float(pred_img)

    valid_gt = gt > thre_depth
    is_close = np.abs(gt - rec) < thre_diff
    mask = np.logical_and(valid_gt, is_close) * 1.0
    len_mask = np.sum(mask)

    grid_x, grid_y = np.meshgrid(
        list(range(res_img[1])),
        list(range(res_img[0]))
    )

    center_x = int(np.sum(grid_x * mask) / len_mask)
    center_y = int(np.sum(grid_y * mask) / len_mask)

    if center_x < center_clip[0]:
        center_x = center_clip[0]
    if center_y < center_clip[1]:
        center_y = center_clip[1]
    if center_x > res_img[0] - center_clip[0]:
        center_x = res_img[0] - center_clip[0]
    if center_y > res_img[1] - center_clip[1]:
        center_y = res_img[1] - center_clip[1]
    
    start_x = center_x - center_clip[1]
    start_y = center_y - center_clip[0]
    end_x = center_x + center_clip[1]
    end_y = center_y + center_clip[0]

    gt_clip = gt_img[start_y: end_y, start_x: end_x, :]
    rec_clip = rec_img[start_y: end_y, start_x: end_x, :]
    shade_clip = shade_img[start_y: end_y, start_x: end_x, :]
    proj_clip = proj_img[start_y: end_y, start_x: end_x, :]
    mask_clip = mask[start_y: end_y, start_x: end_x] * 255
    pred_clip = pred_img[start_y: end_y, start_x: end_x, :]
    pred_depth_clip = pred[start_y: end_y, start_x: end_x]

    # cv2.imwrite(save_gt + '{:05d}.bmp'.format(idx), gt_clip)
    # cv2.imwrite(save_rec + '{:05d}.bmp'.format(idx), rec_clip)
    # cv2.imwrite(save_shade + '{:05d}.png'.format(idx), shade_clip)
    # cv2.imwrite(save_proj + '{:05d}.png'.format(idx), proj_clip)
    # cv2.imwrite(save_mask + '{:05d}.png'.format(idx), mask_clip)
    cv2.imwrite(save_pred + '{:05d}.bmp'.format(idx), pred_clip)
    # np.save(save_pred + '{:05d}.npy'.format(idx), pred_depth_clip)