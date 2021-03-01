import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

DIR = '../data/render/'
DIR = '../data/real/'
# DIR = '../data/input_200318/'

OUT = DIR + 'shading_norm/'

is_shading_norm = True
# is_shading_norm = False

os.makedirs(OUT, exist_ok=True)

for i in range(9):
    # shading = cv2.imread(DIR + 'shade/{:05}.png'.format(i), 0)
    shading = cv2.imread(DIR + 'shade/{:05}.bmp'.format(i), 0)
    # shading = cv2.imread(DIR + 'shading/shading{:03}.bmp'.format(i), 0)

    # print(np.max(shading))
    # print(np.min(shading))

    if is_shading_norm:
        is_shading_available = shading > 16.0
        mask_shading = is_shading_available * 1.0

        # shading = shading / np.max(shading)

        # shading norm : mean 0, var 1
        mean_shading = np.sum(shading) / np.sum(is_shading_available)

        # var_shading = np.var(shading)
        var_shading = np.sum(np.square((shading - mean_shading)*mask_shading)) / np.sum(mask_shading)
        std_shading = np.sqrt(var_shading)

        shading = (shading - mean_shading) / std_shading
        # print(mean_shading)
        # print(std_shading)

        # print(np.max(shading))
        # print(np.min(shading))
        # print(shading[700:705,700:705])
        # shading = shading*127 + 128
        # print(shading[700:705,700:705])

    if is_shading_norm:
        # cv2.imwrite(OUT + '{:05}.png'.format(i), shading*255)
        cv2.imwrite(OUT + '{:05}.png'.format(i), shading*64 + 128)
        # cv2.imwrite(OUT + '{:05}.png'.format(i), (shading*64 + 192)*mask_shading)

        # plt.figure()
        # plt.imshow(shading, cmap='jet', vmin=-2, vmax=2)
        # plt.savefig(OUT + '{:05}.png'.format(i))
    else:
        cv2.imwrite(OUT + '{:05}.png'.format(i), shading)
    