import sys
import numpy as np
import cupy as cp
import cv2
from scipy.spatial.distance import euclidean, cityblock
import time
from tqdm import tqdm

# DIR = 'C:/Users/b19.tokieda/Desktop/cnn-depth_root/'
# src_dir_1 = DIR + 'data/input_200201/'
# src_dir_2 = DIR + 'data/input_200312/'

src_dir = '../data/input_200317/'

src_rec_dir = src_dir + '/rec_ajusted'
src_frame_dir = src_dir + '/frame'
src_gt_dir = src_dir + '/gt'
src_shading_dir = src_dir + '/shading'

src_dir_3 = '../data/input_200318/'

src_rec_dir_3 = src_dir_3 + '/rec_ajusted'
src_frame_dir_3 = src_dir_3 + '/frame'
src_gt_dir_3 = src_dir_3 + '/gt'
src_shading_dir_3 = src_dir_3 + '/shading'

for data_idx in tqdm(range(8)):
    src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
    src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)

    frame = cv2.imread(src_bgra, -1)
    gap = cv2.imread(src_depth_gap, -1)
    gt = cv2.imread(src_depth_gt, -1)
    shading = cv2.imread(src_shading, -1)

    data_idx += 48

    src_bgra_3 = src_frame_dir_3 + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap_3 = src_rec_dir_3 + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt_3 = src_gt_dir_3 + '/gt{:03d}.bmp'.format(data_idx)
    src_shading_3 = src_shading_dir_3 + '/shading{:03d}.bmp'.format(data_idx)

    cv2.imwrite(src_bgra_3, frame)
    cv2.imwrite(src_depth_gap_3, gap)
    cv2.imwrite(src_depth_gt_3, gt)
    cv2.imwrite(src_shading_3, shading)


src_dir = '../data/input_200201-0312/'

src_rec_dir = src_dir + '/rec_ajusted'
src_frame_dir = src_dir + '/frame'
src_gt_dir = src_dir + '/gt'
src_shading_dir = src_dir + '/shading'

for data_idx in tqdm(range(48, 56)):
    src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
    src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)

    frame = cv2.imread(src_bgra, -1)
    gap = cv2.imread(src_depth_gap, -1)
    gt = cv2.imread(src_depth_gt, -1)
    shading = cv2.imread(src_shading, -1)

    data_idx += 8

    src_bgra_3 = src_frame_dir_3 + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap_3 = src_rec_dir_3 + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt_3 = src_gt_dir_3 + '/gt{:03d}.bmp'.format(data_idx)
    src_shading_3 = src_shading_dir_3 + '/shading{:03d}.bmp'.format(data_idx)

    cv2.imwrite(src_bgra_3, frame)
    cv2.imwrite(src_depth_gap_3, gap)
    cv2.imwrite(src_depth_gt_3, gt)
    cv2.imwrite(src_shading_3, shading)


src_dir = '../data/input_200317/'

src_rec_dir = src_dir + '/rec_ajusted'
src_frame_dir = src_dir + '/frame'
src_gt_dir = src_dir + '/gt'
src_shading_dir = src_dir + '/shading'

for data_idx in tqdm(range(8, 17)):
    src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
    src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)

    frame = cv2.imread(src_bgra, -1)
    gap = cv2.imread(src_depth_gap, -1)
    gt = cv2.imread(src_depth_gt, -1)
    shading = cv2.imread(src_shading, -1)

    data_idx += 56

    src_bgra_3 = src_frame_dir_3 + '/frame{:03d}.png'.format(data_idx)
    src_depth_gap_3 = src_rec_dir_3 + '/depth{:03d}.bmp'.format(data_idx)
    src_depth_gt_3 = src_gt_dir_3 + '/gt{:03d}.bmp'.format(data_idx)
    src_shading_3 = src_shading_dir_3 + '/shading{:03d}.bmp'.format(data_idx)

    cv2.imwrite(src_bgra_3, frame)
    cv2.imwrite(src_depth_gap_3, gap)
    cv2.imwrite(src_depth_gt_3, gt)
    cv2.imwrite(src_shading_3, shading)


# frame = cv2.imread(src_dir + 'frame/frame004.png')
# cv2.imwrite(src_dir + 'frame/frame004.png', frame[:, :1200])

# shading = cv2.imread(src_dir + 'shading/shading004.bmp')
# cv2.imwrite(src_dir + 'shading/shading004.bmp', shading[:, :1200])

# argv = sys.argv

# print(argv)

# _, is_model_exist, epoch_num, augment_rate = argv

# print(is_model_exist, epoch_num, augment_rate)


# x = [1, 2, 3]
# y = [4, 5, 6]

# zipped = zip(x, y)

# print(list(zipped))


# img = cp.arange(25, dtype=cp.int16).reshape(5, 5)
# output = cp.zeros((5, 5), dtype=cp.int16)
# F = cp.ones((3, 3)) * 2
# height, width, distance, diameter = 5, 5, 1, 3

# get_smooth_image = cp.ElementwiseKernel(
#         in_params='raw int16 img, uint16 height, uint16 width, uint16 distance, uint16 diameter',
#         out_params='int16 output',
#         preamble=\
#         '''
#         __device__ int get_x_idx(int i, int width) {
#             return i % width;
#         }
#         __device__ int get_y_idx(int i, int height) {
#             return i / height;
#         }
#         __device__ float calc_mean(int i, int x, int y, int height, int width, int distance, int diameter, CArray<int, 1> &img) {
#             int sum = 0;
#             int length = 0;
#             if ( ((x >= distance) && (x < width - distance)) && ((y >= distance) && (y < height - distance)) ) {
#                 for (int k=0; k<diameter; k++) {
#                     for (int l=0; l<diameter; l++) {
#                         sum += img[i + (k-1)*height - distance + l];
#                         length += 1;
#                     }
#                 }
#             }
#             if (sum == 0) {
#                 return 0;
#             } else {
#                 return sum / length;
#             }
#         }
#         ''',
#         operation=\
#         '''
#         int x = get_x_idx(i, width);
#         int y = get_y_idx(i, height);
#         output = calc_mean(i, x, y, height, width, distance, diameter, img);
#         ''',
#         name='get_smooth_image'
# )
# get_smooth_image(img, height, width, distance, diameter, output)
# print(output)