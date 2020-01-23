import depth_tools
import cv2
import numpy as np
from tqdm import tqdm

DIR = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/'
DIR_REC = 'C:/Users/b19.tokieda/Desktop/data_200117/data/rec_rbf50/'
src_dir = DIR + 'input_200117'
src_rec_dir = src_dir + '/rec'
src_gt_dir = src_dir + '/gt'

# input_200117
cam_params = {
    'focal_length': 0.037297750000000005,
    'pix_x': 1.25e-05,
    'pix_y': 1.237130414015908e-05,
    'center_x': 826.396,
    'center_y': 578.887
}

depth_threshold = 0.2
difference_threshold = 0.005

data_idx_range = range(80)

list_thre = [10, 15]
list_bias = [0, 5, -5]

for idx in tqdm(data_idx_range):
    distance = int(idx / 16) * 10 + 80
    rec_idx = idx % 16 + 1

    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(idx)
    img_gt = cv2.imread(src_depth_gt, -1)
    depth_gt = depth_tools.unpack_bmp_bgra_to_float(img_gt)

    is_gt_available = depth_gt > depth_threshold

    list_score = []
    list_comp = []

    for thre in list_thre:
        for bias in list_bias:
            in_rec_dir = DIR_REC + 'output_{}/'.format(thre)
            out_rec_dir = src_dir + '/rec_rbf50/thre_{}_'.format(thre)
            if bias == 0:
                in_rec_dir += 'bias_0/'
                out_rec_dir += 'bias_0/'
            elif bias == 5:
                in_rec_dir += 'bias+5/'
                out_rec_dir += 'bias+5/'
            elif bias == -5:
                in_rec_dir += 'bias-5/'
                out_rec_dir += 'bias-5/'
            
            in_rec_dir += 'output-{}-{}/'.format(distance, rec_idx)
            
            img_rec = cv2.imread(in_rec_dir + 'check13-depthimage.png', -1)
            depth_rec = depth_tools.unpack_png_to_float(img_rec)

            is_depth_close = np.logical_and(
                np.abs(depth_rec - depth_gt) < difference_threshold,
                is_gt_available)

            list_score.append(np.sum(is_depth_close))
            list_comp.append(out_rec_dir)

    best_rec_dir = list_comp[np.argmax(list_score)]
    # best_rec_dir = src_dir + '/rec/'
    best_img_rec = cv2.imread(best_rec_dir + 'depth{:03d}.png'.format(idx), -1)
    cv2.imwrite(src_rec_dir + '/depth{:03d}.png'.format(idx), best_img_rec)

    best_depth_rec = depth_tools.unpack_png_to_float(best_img_rec)
    is_depth_close = np.logical_and(
        np.abs(best_depth_rec - depth_gt) < difference_threshold,
        is_gt_available)
    mask = is_depth_close * 1.0

    xyz_rec = depth_tools.convert_depth_to_coords(best_depth_rec, cam_params)
    xyz_rec_masked = depth_tools.convert_depth_to_coords(best_depth_rec * mask, cam_params)

    depth_tools.dump_ply(src_dir + '/ply_rec/no-mask/rec{:03d}.ply'.format(idx), xyz_rec.reshape(-1, 3).tolist())
    depth_tools.dump_ply(src_dir + '/ply_rec/rec_masked{:03d}.ply'.format(idx), xyz_rec_masked.reshape(-1, 3).tolist())