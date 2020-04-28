import numpy as np
import cv2

import depth_tools

DIR = '../output/output_vloss_aug_drop30/predict_100/'
# DIR = '../data/input_200318/rec_ajusted/'
# DIR = '../data/render_h005/rec/'

img = cv2.imread(DIR + 'predict-161.bmp', -1)
depth = depth_tools.unpack_bmp_bgra_to_float(img)
np.save(DIR + 'pred.npy', depth)