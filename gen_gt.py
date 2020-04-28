import cv2
from tqdm import tqdm
import time

import common_tools as tool
import depth_tools

DIR = '../data/input_200317/'
DIR = '../data/data_200427/'
OUT = '../data/real/'

idx = range(9)

# calib 200427
cam_params = {
    'focal_length': 0.036917875,
    'pix_x': 1.25e-05,
    'pix_y': 1.2416172558410155e-05,
    'center_x': 785.81,
    'center_y': 571.109
}

time_start = time.time()

for i in tqdm(idx):
    gt_ori = cv2.imread(DIR + 'gt/{:05d}.bmp'.format(i), -1)
    gt_mask = cv2.imread(DIR + 'mask_gt/{:05d}.bmp'.format(i), -1)

    gt_ori = gt_ori[:, :1200, :]
    gt_mask = gt_mask[:, :1200]

    gt_img = tool.delete_mask(gt_ori, gt_mask)
    gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

    new_gt = tool.gaussian_filter(gt, 4)

    new_gt_img = depth_tools.pack_float_to_bmp_bgra(new_gt)
    cv2.imwrite(OUT + 'gt/{:05d}.bmp'.format(i), new_gt_img)

    xyz_gt = depth_tools.convert_depth_to_coords(new_gt, cam_params)
    depth_tools.dump_ply(OUT + 'ply_gt/{:05d}.ply'.format(i), xyz_gt.reshape(-1, 3).tolist())

time_end = time.time()
print('{:.2f} sec'.format(time_end - time_start))