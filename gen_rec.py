import depth_tools
import cv2
import numpy as np
from tqdm import tqdm
import os

DIR = '../'
DIR_REC = 'C:/Users/b19.tokieda/Desktop/data_200317/reconst/'
src_dir = DIR + 'data/input_200317'
src_rec_dir = src_dir + '/rec'
src_gt_dir = src_dir + '/gt'

# input_200117
# cam_params = {
#     'focal_length': 0.037297750000000005,
#     'pix_x': 1.25e-05,
#     'pix_y': 1.237130414015908e-05,
#     'center_x': 826.396,
#     'center_y': 578.887
# }
# input_200201
cam_params = {
    'focal_length': 0.037306625,
    'pix_x': 1.25e-05,
    'pix_y': 1.2360472397638345e-05,
    'center_x': 801.557,
    'center_y': 555.618
}

depth_threshold = 0.2
difference_threshold = 0.005

data_idx_range = range(17)

list_thre = [10, 15]
list_bias = [0, 5, -5]
# list_bias = [0, 5]

for idx in tqdm(data_idx_range):
    # if idx < 40:
    #     datatype = 'b'
    #     distance = int(idx / 8) * 10 + 90
    #     rec_idx = idx % 8 + 1
    # else:
    #     datatype = 'f'
    #     distance = int((idx - 40) / 4) * 10 + 90
    #     rec_idx = (idx - 40) % 4 + 1

    if idx < 4:
        datatype = 'mid'
        distance = 90
        rec_idx = idx + 1
    elif idx < 8:
        datatype = 'mid'
        distance = 100
        rec_idx = idx - 3
    elif idx < 12:
        datatype = 'mid'
        distance = 130
        rec_idx = idx - 7
    elif idx < 17:
        datatype = 'spo'
        rec_idx = 1
        if idx == 12:
            distance = 90
        elif idx == 13:
            distance = 100
        elif idx == 14:
            distance = 110
        elif idx == 15:
            distance = 120
        elif idx == 16:
            distance = 130

    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(idx)
    img_gt = cv2.imread(src_depth_gt, -1)
    depth_gt = depth_tools.unpack_bmp_bgra_to_float(img_gt)

    is_gt_available = depth_gt > depth_threshold

    list_score = []
    list_comp = []

    for thre in list_thre:
        for bias in list_bias:
            in_rec_dir = DIR_REC + 'output_{}-{}-{}_{}-{}/'.format(datatype, distance, rec_idx, thre, bias)
            out_rec_dir = src_dir + '/rec_original/rec_{}_{}/'.format(thre, bias)
            os.makedirs(out_rec_dir, exist_ok=True)
            
            img_rec = cv2.imread(in_rec_dir + 'check13-depthimage.png', -1)
            cv2.imwrite(out_rec_dir + 'depth{:03d}.png'.format(idx), img_rec)
            depth_rec = depth_tools.unpack_png_to_float(img_rec)

            is_depth_close = np.logical_and(
                np.abs(depth_rec - depth_gt) < difference_threshold,
                is_gt_available)

            list_score.append(np.sum(is_depth_close))
            list_comp.append(out_rec_dir)

    best_rec_dir = list_comp[np.argmax(list_score)]
    best_img_rec = cv2.imread(best_rec_dir + 'depth{:03d}.png'.format(idx), -1)
    cv2.imwrite(src_rec_dir + '/depth{:03d}.png'.format(idx), best_img_rec)

    best_depth_rec = depth_tools.unpack_png_to_float(best_img_rec)
    is_depth_close = np.logical_and(
        np.abs(best_depth_rec - depth_gt) < difference_threshold,
        is_gt_available)
    mask = is_depth_close * 1.0

    best_depth_rec_masked = best_depth_rec * mask

    xyz_rec = depth_tools.convert_depth_to_coords(best_depth_rec, cam_params)
    xyz_rec_masked = depth_tools.convert_depth_to_coords(best_depth_rec_masked, cam_params)

    depth_tools.dump_ply(src_dir + '/ply_rec/rec{:03d}.ply'.format(idx), xyz_rec.reshape(-1, 3).tolist())
    depth_tools.dump_ply(src_dir + '/ply_rec_masked/rec_masked{:03d}.ply'.format(idx), xyz_rec_masked.reshape(-1, 3).tolist())