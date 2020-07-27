import depth_tools
import cv2
import numpy as np
from tqdm import tqdm

DIR = '../'
# DIR = 'C:/Users/Kodai Tokieda/Desktop/PhotometricAugmentation/PhotometricAugmentation/procam_rasterizer/'
# data_dir = DIR + 'data/input_200317/'
data_dir = DIR + 'data/render/'
# data_dir = DIR + 'render/'
rec_dir = data_dir + 'rec_ori/'
gt_dir = data_dir + 'gt/'

data_num = 200

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

depth_threshold = 0.2
difference_threshold = 0.01

data_idx_range = range(data_num)

for idx in tqdm(data_idx_range):
    img_gt = cv2.imread(gt_dir + '{:05d}.bmp'.format(idx), -1)
    img_rec = cv2.imread(rec_dir + '{:05d}.bmp'.format(idx), -1)
    depth_gt = depth_tools.unpack_bmp_bgra_to_float(img_gt)
    # depth_rec = depth_tools.unpack_png_to_float(img_rec)
    depth_rec = depth_tools.unpack_bmp_bgra_to_float(img_rec)

    is_gt_available = depth_gt > depth_threshold
    is_depth_close = np.logical_and(
                np.abs(depth_rec - depth_gt) < difference_threshold,
                is_gt_available)
    mask = is_depth_close * 1.0

    depth_gt_masked = depth_gt * mask
    depth_rec_masked = depth_rec * mask

    length = np.sum(mask)

    # mean_gt = np.sum(depth_gt_masked) / length
    # mean_rec = np.sum(depth_rec_masked) / length
    # gap = mean_gt - mean_rec
    # depth_rec_ajust = np.where(depth_rec > depth_threshold, depth_rec + gap, 0)


    gt_diff = depth_gt_masked - depth_rec_masked
    mean_gt_diff = np.sum(gt_diff) / length
    depth_rec_ajust = (depth_rec_masked + mean_gt_diff) * mask


    img_rec_ajust = depth_tools.pack_float_to_bmp_bgra(depth_rec_ajust)
    cv2.imwrite(data_dir + 'rec/{:05d}.bmp'.format(idx), img_rec_ajust)

    # xyz_rec = depth_tools.convert_depth_to_coords(depth_rec_ajust, cam_params)
    # depth_tools.dump_ply(data_dir + 'ply_rec_ajust_old/{:05d}.ply'.format(idx), xyz_rec.reshape(-1, 3).tolist())


    # is_depth_close = np.logical_and(
    #             np.abs(depth_rec_ajust - depth_gt) < difference_threshold,
    #             is_gt_available)
    # mask = is_depth_close * 1.0

    # depth_rec_ajust_masked = depth_rec_ajust * mask
    # img_rec_ajust_masked = depth_tools.pack_float_to_bmp_bgra(depth_rec_ajust_masked)
    # cv2.imwrite(data_dir + 'rec_ajusted_masked/{:05d}.bmp'.format(idx), img_rec_ajust_masked)
    # xyz_rec_masked = depth_tools.convert_depth_to_coords(depth_rec_ajust_masked, cam_params)
    # depth_tools.dump_ply(data_dir + '/ply_rec_ajusted_masked/{:05d}.ply'.format(idx), xyz_rec_masked.reshape(-1, 3).tolist())
