import cv2
import numpy as np
import matplotlib.pyplot as plt
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
import torchvision.transforms as transforms

import depth_tools
import network_pytorch as network

#InputData
# src_dir = '../data/input_200318'
src_dir = '../data/render'
#OutputDir
out_dir = '../output/pytorch_test'

'''data index'''
# train_idx_range = list(range(16)) # 0 - 15
# train_idx_range.extend(list(range(24, 40))) # 24 - 43
# train_idx_range.extend(list(range(40, 44)))
# # train_idx_range.extend(list(range(44, 48)))
# train_idx_range.extend(list(range(48, 56))) # 48 - 55
# train_idx_range.extend(list(range(60, 68))) # 60 -67
# # train_idx_range.extend(list(range(68, 73)))

train_idx_range = list(range(40))
test_idx_range = list(range(40, 50))

# train_idx_range = list(range(80))
# test_idx_range = list(range(80, 100))

# train_idx_range = list(range(8))
# test_idx_range = list(range(8, 10))

# train_idx_range = list(range(160))
# test_idx_range = list(range(160, 200))

epochs = 100

# parameters
depth_threshold = 0.2
difference_threshold = 0.005
patch_remove = 0.5
dropout_rate = 0.1

img_shape = (1200, 1200)
patch_shape = (120, 120)
patch_tl = (0, 0)  # top, left
num_patch = 100

# clip patches
p_top, p_left = patch_tl
i_h, i_w = img_shape
p_h, p_w = patch_shape
top_coords = range(p_top, i_h, p_h)
left_coords = range(p_left, i_w, p_w)

ch_num = 2
train_batch_size = 100
test_batch_size = 100
num_print_in_epoch = 2
num_batch_in_img = num_patch / test_batch_size

difference_scaling = 1

os.makedirs(out_dir, exist_ok=True)

def prepare_data(train_idx_range, train=True):
    def clip_patch(img, top_left, size):
        t, l, h, w = *top_left, *size
        # t = top_left[0] # for Python2
        # l = top_left[1]
        # h = size[0]
        # w = size[1]
        return img[t:t + h, l:l + w]

    # data dir
    src_depth_dir = src_dir + '/depth'
    src_proj_dir = src_dir + '/proj'
    src_shade_dir = src_dir + '/shade'

    # read data
    print('loading data...')
    x_train = []
    y_train = []
    for data_idx in tqdm(train_idx_range):
        # data name
        src_depth = src_depth_dir + '/{:05d}.bmp'.format(data_idx)
        src_proj = src_proj_dir + '/{:05d}.png'.format(data_idx)
        src_shade = src_shade_dir + '/{:05d}.png'.format(data_idx)

        # read data
        depth_img = cv2.imread(src_depth, -1)
        proj = cv2.imread(src_proj, 0) / 255.
        shade = cv2.imread(src_shade, 0) / 255.
        
        # depth from image
        depth = depth_tools.unpack_bmp_bgra_to_float(depth_img)

        is_depth_available = depth > depth_threshold
        mask = is_depth_available * 1.0

        # depth -> 0
        mean_depth = np.sum(depth) / np.sum(mask)
        depth -= mean_depth
        depth *= mask

        # difference scaling
        # print('')
        # print(np.max(depth))
        # print(np.min(depth))
        depth *= difference_scaling
        # print('')
        # print(np.max(depth))
        # print(np.min(depth))

        x_img = np.dstack([proj, shade])
        y_img = np.dstack([depth, mask])

        # add training data
        for top, left in product(top_coords, left_coords):
            patch_x = clip_patch(x_img, (top, left), patch_shape)
            patch_y = clip_patch(y_img, (top, left), patch_shape)

            # if train:
            #     if np.sum(patch_y[:, :, 1]) < patch_shape[0] * patch_shape[1] * patch_remove:
            #         continue
            
            x_train.append(patch_x.transpose(2, 0, 1))
            y_train.append(patch_y.transpose(2, 0, 1))
    return np.array(x_train), np.array(y_train)

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

def loss_graph(list_loss):
    list_x = range(len(list_loss))

    plt.figure()
    plt.plot(list_x, list_loss)
    plt.savefig(out_dir + '/loss.pdf')

def train(device):
    x_data, y_data = prepare_data(train_idx_range)
    print('x train shape:', x_data.shape)
    print('y train shape:', y_data.shape)

    len_data = len(x_data)
    num_batch = len_data // train_batch_size
    len_running = num_batch // num_print_in_epoch

    x_train = torch.from_numpy(x_data).float()
    y_train = torch.from_numpy(y_data).float()

    trainset = TensorDataset(x_train, y_train)
    trainloader = DataLoader(
        trainset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=2, 
        # collate_fn=my_collate_fn
    )

    model = network.UNet(ch_num, ch_num)
    model.to(device)

    # define loss function and optimier
    # criterion = mean_squared_error()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters()) #default lr=0.001

    list_loss = []

    # train
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, gts) in enumerate(trainloader, 0):
            inputs, gts = torch.Tensor(inputs).to(device), torch.Tensor(gts).to(device)
            # print('inputs', inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print('outputs', outputs.shape)
            output, gt = outputs[:100, 0, :, :], gts[:, 0, :, :]
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % len_running == 0:
                tmp_loss = running_loss/len_running
                list_loss.append(tmp_loss)
                print('[{:5d}, {:5d}] loss: {:.5f}'
                      .format(epoch+1, i+1, tmp_loss))
                running_loss = 0.0

    print('Finished Training')
    loss_graph(list_loss)
    return model

def predict(device, model):
    def merge_patch(output_batch):
        predict = torch.from_numpy(np.zeros(img_shape))
        j = 0
        for top, left in product(top_coords, left_coords):
            t, l, h, w = top, left, *patch_shape
            predict[t:t + h, l:l + w] = output_batch[j, :, :]
            j += 1
        return predict.numpy()

    x_test, y_test = prepare_data(test_idx_range, train=False)
    print('x test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    testset = TensorDataset(x_test, y_test)
    testloader = DataLoader(
        testset, 
        batch_size=test_batch_size, 
        shuffle=False,
        num_workers=2
    )

    criterion = nn.MSELoss()

    losses = 0.0
    test_loss = 0.0
    cnt = 0
    list_predict = []

    with torch.no_grad():
        for i, (inputs, gts) in enumerate(testloader, 0):
            inputs, gts = torch.Tensor(inputs).to(device), torch.Tensor(gts).to(device)
            outputs = model(inputs)
            output, gt = outputs[:100, 0, :, :], gts[:, 0, :, :]
            # print(outputs.shape)
            # print(output.shape)
            loss = criterion(output, gt)
            losses += loss
            test_loss += loss
            cnt += 1
            if (i + 1) % num_batch_in_img == 0:
                print('{:5d} loss: {:.5f}'.format(i + 1, losses / num_batch_in_img))
                losses = 0.0

                pred = merge_patch(output)
                pred_bmp = depth_tools.pack_float_to_bmp_bgra(pred)
                cv2.imwrite(out_dir + '/pred{:03d}.bmp'.format(test_idx_range[i]), pred_bmp)
                list_predict.append(pred)

    test_loss /= cnt
    print('Test loss: {:.5f}'.format(test_loss))
    return test_loss

def main(device):
    model = train(device)
    test_loss = predict(device, model)

if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))