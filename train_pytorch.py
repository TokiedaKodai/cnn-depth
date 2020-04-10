import depth_tools
import cv2
import numpy as np
from itertools import product
import os
from tqdm import tqdm
from glob import glob
import re
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from utils.dataset import BasicDataset

import network_pytorch as network

#InputData
src_dir = '../data/input_200318'
#OutputDir
# out_dir = 'output'
# out_dir = '../output/' + out_local
out_dir = '../output/output_test'

'''data index'''
data_idx_range = list(range(16)) # 0 - 15
# data_idx_range.extend(list(range(24, 40))) # 24 - 43
# data_idx_range.extend(list(range(40, 44)))
# # data_idx_range.extend(list(range(44, 48)))
# data_idx_range.extend(list(range(48, 56))) # 48 - 55
# data_idx_range.extend(list(range(60, 68))) # 60 -67
# # data_idx_range.extend(list(range(68, 73)))

epochs = 10

# parameters
depth_threshold = 0.2
difference_threshold = 0.005
patch_remove = 0.5
dropout_rate = 0.1

batch_shape = (120, 120)
batch_tl = (0, 0)  # top, left

ch_num = 3
train_batch_size = 64

difference_scaling = 100

is_shading_norm = True

def prepare_data(data_idx_range):
    def clip_batch(img, top_left, size):
        # t, l, h, w = *top_left, *size
        t = top_left[0]
        l = top_left[1]
        h = size[0]
        w = size[1]
        return img[t:t + h, l:l + w]

    # src_rec_dir = src_dir + '/rec'
    src_rec_dir = src_dir + '/rec_ajusted'
    src_frame_dir = src_dir + '/frame'
    src_gt_dir = src_dir + '/gt'
    src_shading_dir = src_dir + '/shading'

    # read data
    print('loading data...')
    x_train = []
    y_train = []
    for data_idx in tqdm(data_idx_range):
        src_bgra = src_frame_dir + '/frame{:03d}.png'.format(data_idx)
        # src_depth_gap = src_rec_dir + '/depth{:03d}.png'.format(data_idx)
        src_depth_gap = src_rec_dir + '/depth{:03d}.bmp'.format(data_idx)
        src_depth_gt = src_gt_dir + '/gt{:03d}.bmp'.format(data_idx)
        # src_shading = src_shading_dir + '/shading{:03d}.png'.format(data_idx)
        src_shading = src_shading_dir + '/shading{:03d}.bmp'.format(data_idx)

        # read images
        bgr = cv2.imread(src_bgra, -1) / 255.
        depth_img_gap = cv2.imread(src_depth_gap, -1)
        # depth_gap = depth_tools.unpack_png_to_float(depth_img_gap)
        depth_gap = depth_tools.unpack_bmp_bgra_to_float(depth_img_gap)

        depth_img_gt = cv2.imread(src_depth_gt, -1)
        depth_gt = depth_tools.unpack_bmp_bgra_to_float(depth_img_gt)
        img_shape = bgr.shape[:2]

        shading_bgr = cv2.imread(src_shading, -1)
        # shading = cv2.imread(src_shading, 0) # GrayScale
        shading = np.zeros_like(shading_bgr)
        shading[:, :, 0] = 0.299 * shading_bgr[:, :, 2] + 0.587 * shading_bgr[:, :, 1] + 0.114 * shading_bgr[:, :, 0]

        if is_shading_norm:
            shading = shading / np.max(shading)
        else:
            shading = shading / 255.


        # normalization (may not be needed)
        # depth_gap /= depth_gap.max()
        # depth_gt /= depth_gt.max()

        depth_thre = depth_threshold

        # merge bgr + depth_gap
        bgrd = np.dstack([shading[:, :, 0], depth_gap, bgr[:, :, 0]])

        # clip batches
        b_top, b_left = batch_tl
        b_h, b_w = batch_shape
        top_coords = range(batch_tl[0], img_shape[0], batch_shape[0])
        left_coords = range(batch_tl[1], img_shape[1], batch_shape[1])

        # add training data
        for top, left in product(top_coords, left_coords):
            batch_train = clip_batch(bgrd, (top, left), batch_shape)

            # do not add batch if not valid ################
            # valid_pixels = np.logical_and(
            #     batch_train[:, :, 0].mean() > 0,
            #     batch_train[:, :, 1] > depth_threshold)
            # if np.count_nonzero(valid_pixels) < (b_h * b_w * 0.5):
            #     continue

            batch_gt_depth = clip_batch(depth_gt, (top, left), batch_shape)
            batch_gt_mask = clip_batch(depth_gap, (top, left), batch_shape)
            # batch_gt = np.dstack([batch_gt_depth, batch_gt_mask])
            batch_gt = np.dstack([batch_gt_depth])

            # do not add batch if not close ################
            is_gt_available = batch_gt_depth > depth_thre
            is_depth_close = np.logical_and(
                np.abs(batch_train[:, :, 1] - batch_gt_depth) < difference_threshold,
                is_gt_available)
            if np.count_nonzero(is_depth_close) < (b_h * b_w * patch_remove):
                continue
            
            batch_train = batch_train.transpose(2, 0, 1)
            # batch_gt = batch_gt.reshape((batch_shape[0], batch_shape[1], 2))
            batch_gt = batch_gt.reshape((batch_shape[0], batch_shape[1], 1))
            batch_gt = batch_gt.transpose(2, 0, 1)

            x_train.append(batch_train)
            y_train.append(batch_gt)
    return np.array(x_train), np.array(y_train), depth_thre

def mean_squared_error(predict, gt):
    depth_gt = gt[:, :, :, 0]
    depth_gap = gt[:, :, :, 1]

    is_gt_available = depth_gt > depth_threshold
    is_gap_unavailable = depth_gap < depth_threshold

    # is_depth_close = torch.from_numpy(np.logical_and(np.stack([
    #     np.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available])))
    is_depth_close = torch.from_numpy(np.logical_and(
        np.abs(depth_gap - depth_gt) < difference_threshold, is_gt_available))

    # difference learn
    gt = depth_gt - depth_gap

    # scale
    gt = gt * difference_scaling

    is_valid = is_depth_close.float()

    valid_length = torch.sum(is_valid)
    err = torch.sum(torch.pow((gt - predict[:, :, :, 0]) * is_valid, 2))
    return err / valid_length

def my_collate_fn(batch):
    # # datasetの出力が
    # # [image, target] = dataset[batch_idx]
    # # の場合.
    # images = []
    # targets = []
    # for sample in batch:
    #     image, target = sample
    #     images.append(image)
    #     targets.append(target)
    # images = torch.stack(images, dim=0)
    # return images, targets
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    targets = torch.stack(targets)
    return images, targets

def main(device):
    x_data, y_data, depth_thre = prepare_data(data_idx_range)
    print('x train data:', x_data.shape)
    print('y train data:', y_data.shape)

    x_train = torch.from_numpy(x_data).float()
    y_train = torch.from_numpy(y_data).float()

    # trainset = TensorDataset(x_train, y_train)
    trainset = BasicDataset(x_train, y_train)
    trainloader = DataLoader(
        trainset, 
        batch_size=train_batch_size, 
        shuffle=True, num_workers=2, 
        collate_fn=my_collate_fn
    )

    model = network.UNet(ch_num, 1)
    model.to(device)

    # define loss function and optimier
    # criterion = mean_squared_error()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # train
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, gts) in enumerate(trainloader, 0):
            inputs, gts = torch.Tensor(inputs).to(device), torch.Tensor(gts).to(device)
            # inputs, gts = Variable(inputs), Variable(gts)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, gts)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 40 == 39:
            #     print('[{:d}, {:5d}] loss: {:.3f}'
            #           .format(epoch+1, i+1, running_loss/40))
            #     running_loss = 0.0
            print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch+1, i+1, loss.item()))

    print('Finished Training')


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))