import sys
import numpy as np
import cupy as cp
import cv2
from scipy.spatial.distance import euclidean, cityblock
import time

DIR = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/'
src_dir = DIR + 'input_data_1217/'

frame = cv2.imread(src_dir + 'frame/frame004.png')
cv2.imwrite(src_dir + 'frame/frame004.png', frame[:, :1200])

shading = cv2.imread(src_dir + 'shading/shading004.bmp')
cv2.imwrite(src_dir + 'shading/shading004.bmp', shading[:, :1200])

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