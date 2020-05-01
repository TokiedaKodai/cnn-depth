import cv2
from tqdm import tqdm
import time
import os

import common_tools as tool
import depth_tools

# DIR = '../output/'
DIR = '../data/input_200318/'

idx = list(range(16, 24))
idx.extend(list(range(44, 48)))
idx.extend(list(range(56, 60)))

idx = range(73)

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

for i in tqdm(idx):
    img_depth = cv2.imread(DIR + 'rec_ajusted/depth{:03d}.bmp'.format(i), -1)
    depth = depth_tools.unpack_bmp_bgra_to_float(img_depth)
    xyz_depth = depth_tools.convert_depth_to_coords(depth, cam_params)
    depth_tools.dump_ply(DIR + 'ply_rec/rec{:03d}.ply'.format(i), xyz_depth.reshape(-1, 3).tolist())