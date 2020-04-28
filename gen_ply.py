import cv2
from tqdm import tqdm
import time
import os

import common_tools as tool
import depth_tools

DIR = '../output'
OUTPUT = DIR + '/output_vloss_aug_drop10'
PRED = OUTPUT + '/predict_300_board'
PLY = PRED + '/ply'

idx = list(range(16, 24))
idx.extend(list(range(44, 48)))
idx.extend(list(range(56, 60)))

# calib 200427
cam_params = {
    'focal_length': 0.036917875,
    'pix_x': 1.25e-05,
    'pix_y': 1.2416172558410155e-05,
    'center_x': 785.81,
    'center_y': 571.109
}

os.makedirs(PLY, exist_ok=True)

for i in tqdm(idx):
    img_depth = cv2.imread(PRED + '/predict-{:03d}.bmp'.format(i), -1)
    depth = depth_tools.unpack_bmp_bgra_to_float(img_depth)
    xyz_depth = depth_tools.convert_depth_to_coords(depth, cam_params)
    depth_tools.dump_ply(PLY + '/{:05d}.ply'.format(i), xyz_depth.reshape(-1, 3).tolist())