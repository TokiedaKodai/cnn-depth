import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
from tqdm import tqdm

do_cut_gt = False
# isTrain = True

root_dir = './'
# input_dir = root_dir + 'input_data_1023/'
input_dir = root_dir + 'input_data_1119/'
output_dir = root_dir + 'augment_data_1119/'

original_data_index = range(24)
data_index = list(range(6))
data_index.extend(list(range(12, 24)))
augument_rate = 4

# if isTrain:
#     # Train Data
#     data_index = range(0, 100)
#     augument_rate = 4
#     output_dir = root_dir + 'augment_train/'
# else:
#     # Test Data
#     data_index = range(100, 120)
#     augument_rate = 2
#     output_dir = root_dir + 'augment_test/'


in_frame_dir = input_dir + 'frame/'
in_rec_dir = input_dir + 'rec/'
in_gt_dir = input_dir + 'gt/'
# in_shading_dir = input_dir + 'shading/'
in_shading_dir = input_dir + 'shading_norm/'

out_frame_dir = output_dir + 'frame/'
out_rec_dir = output_dir + 'rec/'
out_gt_dir = output_dir + 'gt/'
# out_shading_dir = output_dir + 'shading/'
out_shading_dir = output_dir + 'shading_norm/'

# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(out_frame_dir, exist_ok=True)
# os.makedirs(out_rec_dir, exist_ok=True)
# os.makedirs(out_gt_dir, exist_ok=True)
# os.makedirs(out_shading_dir, exist_ok=True)

def cut_gt():
    n = 0
    while os.path.exists(in_gt_dir + 'gt%05d.bmp'%n):
        gt = cv2.imread(in_gt_dir + 'gt%05d.bmp'%n, -1)
        gt = gt[0:1200, 0:1200, :]
        cv2.imwrite(in_gt_dir + 'gt%05d.bmp'%n, gt)
        n += 1

if do_cut_gt:
    cut_gt()


datagen = ImageDataGenerator(
        # rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0,
        zoom_range=[0.8, 1.2],
        fill_mode='constant',
        cval=0,
        # horizontal_flip=True,
        # vertical_flip=True
        )

cnt = 0

# original data
for i in tqdm(original_data_index):
#     img_frame = cv2.imread(in_frame_dir + 'frame%05d.bmp'%i, -1)
    img_frame = cv2.imread(in_frame_dir + 'frame%05d.png'%i, -1)
    img_rec = cv2.imread(in_rec_dir + 'depth%05d.png'%i, -1)
    img_gt = cv2.imread(in_gt_dir + 'gt%05d.bmp'%i, -1)
#     img_shading = cv2.imread(in_shading_dir + 'shading%05d.png'%i, -1)
    img_shading = cv2.imread(in_shading_dir + 'shading_norm%05d.png'%i, -1)

    cv2.imwrite(out_frame_dir + 'frame%05d.bmp'%cnt, img_frame)
    cv2.imwrite(out_rec_dir + 'depth%05d.png'%cnt, img_rec)
    cv2.imwrite(out_gt_dir + 'gt%05d.bmp'%cnt, img_gt)
#     cv2.imwrite(out_shading_dir + 'shading%05d.png'%cnt, img_shading)
    cv2.imwrite(out_shading_dir + 'shading_norm%05d.png'%cnt, img_shading)
    cnt +=1


for i in tqdm(data_index):
#     img_frame = cv2.imread(in_frame_dir + 'frame%05d.bmp'%i, -1)
    img_frame = cv2.imread(in_frame_dir + 'frame%05d.png'%i, -1)
    img_rec = cv2.imread(in_rec_dir + 'depth%05d.png'%i, -1)
    img_gt = cv2.imread(in_gt_dir + 'gt%05d.bmp'%i, -1)
#     img_shading = cv2.imread(in_shading_dir + 'shading%05d.png'%i, -1)
    img_shading = cv2.imread(in_shading_dir + 'shading_norm%05d.png'%i, -1)

    seed = i

    img = img_frame
    img = img.reshape((1,) + img.shape)
    gen_frame = datagen.flow(img, batch_size=1, seed=seed)

    img = img_rec
    img = img.reshape((1,) + img.shape)
    gen_rec = datagen.flow(img, batch_size=1, seed=seed)

    img = img_gt
    img = img.reshape((1,) + img.shape)
    gen_gt = datagen.flow(img, batch_size=1, seed=seed)

    img = img_shading
    img = img.reshape((1,) + img.shape)
    gen_shading = datagen.flow(img, batch_size=1, seed=seed)

    for j in range(augument_rate):
        batch_frame = next(gen_frame)
        batch_rec = next(gen_rec)
        batch_gt = next(gen_gt)
        batch_shading = next(gen_shading)

        gen_img_frame = batch_frame[0]
        gen_img_rec = batch_rec[0]
        gen_img_gt = batch_gt[0]
        gen_img_shading = batch_shading[0]

        # cv2.imwrite(out_frame_dir + 'frame%05d.bmp'%cnt, gen_img_frame)
        cv2.imwrite(out_frame_dir + 'frame%05d.png'%cnt, gen_img_frame)
        cv2.imwrite(out_rec_dir + 'depth%05d.png'%cnt, gen_img_rec)
        cv2.imwrite(out_gt_dir + 'gt%05d.bmp'%cnt, gen_img_gt)
        # cv2.imwrite(out_shading_dir + 'shading%05d.png'%cnt, gen_img_shading)
        cv2.imwrite(out_shading_dir + 'shading_norm%05d.png'%cnt, gen_img_shading)
        cnt += 1
