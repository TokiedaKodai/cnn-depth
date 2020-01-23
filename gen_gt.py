import cv2
from tqdm import tqdm
import time

import common_tools as tool
import depth_tools

DIR = 'input_200117/'

idx = range(80)

# input_200117
cam_params = {
    'focal_length': 0.037297750000000005,
    'pix_x': 1.25e-05,
    'pix_y': 1.237130414015908e-05,
    'center_x': 826.396,
    'center_y': 578.887
}

time_start = time.time()

for i in tqdm(idx):
    gt_ori = cv2.imread(DIR + 'gt-original/gt{:03d}.bmp'.format(i), -1)
    gt_mask = cv2.imread(DIR + 'gt-mask/mask{:03d}.bmp'.format(i), -1)

    gt_img = tool.delete_mask(gt_ori, gt_mask)
    gt = depth_tools.unpack_bmp_bgra_to_float(gt_img)

    new_gt = tool.gaussian_filter(gt, 4)

    new_gt_img = depth_tools.pack_float_to_bmp_bgra(new_gt)
    cv2.imwrite(DIR + 'gt/gt{:03d}.bmp'.format(i), new_gt_img)

    xyz_gt = depth_tools.convert_depth_to_coords(new_gt, cam_params)
    depth_tools.dump_ply(DIR + 'ply_gt/gt{:03d}.ply'.format(i), xyz_gt.reshape(-1, 3).tolist())

time_end = time.time()
print('{:.2f} sec'.format(time_end - time_start))