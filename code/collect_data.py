import cv2

local_dir = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/'
data_dir = 'C:/Users/b19.tokieda/Desktop/data_200117/data/'
output_dir = local_dir + 'input_200117/'


def collect(datatype, distance, idx, depth_cnt, cnt):
    if datatype is 'b':
        folder = data_dir + 'board/{}-{}-{}/'.format(datatype, distance, idx)
    elif datatype is 'f':
        folder = data_dir + 'fake/{}-{}-{}/'.format(datatype, distance, idx)
    print(folder)
    # gt_img = cv2.imread(folder + 'NewScan000.depth.dist.bmp', -1)
    # frame_img = cv2.imread(folder + 'frame00000.png', -1)
    shading_img = cv2.imread(folder + 'frame00010.png', -1)
    # gt_mask = cv2.imread(folder + 'NewScan000.depth.mask.bmp', -1)
    # depth_dir = data_dir + 'rec_rbf40/output_10/bias_0/output-{}-{}/'.format(distance, depth_cnt)
    # depth_img = cv2.imread(depth_dir + 'check13-depthimage.png', -1)

    # cv2.imwrite(output_dir + 'gt/gt%03d.bmp'%cnt, gt_img[:1200, :1200, :])
    # cv2.imwrite(output_dir + 'frame/frame%03d.png'%cnt, frame_img[:1200, :1200, :])
    cv2.imwrite(output_dir + 'shading/shading%03d.png'%cnt, shading_img[:1200, :1200, :])
    # cv2.imwrite(output_dir + 'gt_mask/mask%03d.bmp'%cnt, gt_mask[:1200, :1200])
    # cv2.imwrite(output_dir + 'rec/depth%03d.png'%cnt, depth_img)

    # print(' %02d'%cnt, end='\r')


list_distance = [80, 90, 100, 110, 120]
list_datatype = ['b', 'f']
list_threshold = [10, 15]

cnt = 0

for distance in list_distance:
    for datatype in list_datatype:
        if datatype is 'b':
            for idx in range(1, 13):
                depth_cnt = idx
                collect(datatype, distance, idx, depth_cnt, cnt)
                cnt += 1
        elif datatype is 'f':
            for idx in range(1, 5):
                depth_cnt = idx + 12
                collect(datatype, distance, idx, depth_cnt, cnt)
                cnt += 1
        