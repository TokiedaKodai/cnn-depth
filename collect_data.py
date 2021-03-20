import cv2

root_dir = '/cnn-depth-root/'
data_dir = '/data_200317/'
output_dir = root_dir + 'data/input_200317/'

def collect(datatype, distance, idx, cnt):
    folder = data_dir + '{}-{}-{}/'.format(datatype, distance, idx)

    gt_img = cv2.imread(folder + 'NewScan000.depth.dist.bmp', -1)
    frame_img = cv2.imread(folder + 'frame00000.png', -1)
    shading_img_png = cv2.imread(folder + 'frame00011.png', -1)
    shading_img = cv2.imread(folder + 'NewScan000.depth.texture000.img.bmp', -1)
    gt_mask = cv2.imread(folder + 'NewScan000.depth.mask.bmp', -1)

    cv2.imwrite(output_dir + 'gt_original/gt%03d.bmp'%cnt, gt_img[:1200, :1200, :])
    cv2.imwrite(output_dir + 'frame/frame%03d.png'%cnt, frame_img[:1200, :1200, :])
    cv2.imwrite(output_dir + 'shading_png/shading%03d.png'%cnt, shading_img_png[:1200, :1200, :])
    cv2.imwrite(output_dir + 'shading/shading%03d.bmp'%cnt, shading_img[:1200, :1200, :])
    cv2.imwrite(output_dir + 'gt_mask/mask%03d.bmp'%cnt, gt_mask[:1200, :1200])

    print(' %02d'%cnt, end='\r')


list_distance = [90, 100, 110, 120, 130]
list_datatype = ['b', 'f']

cnt = 0

# for datatype in list_datatype:
#     for distance in list_distance:
#         if datatype is 'b':
#             for idx in range(1, 9):
#                 collect(datatype, distance, idx, cnt)
#                 cnt += 1
#         elif datatype is 'f':
#             for idx in range(1, 5):
#                 collect(datatype, distance, idx, cnt)
#                 cnt += 1
        
# datatype = 'small'
# distance = 90
# for idx in range(1, 5):
#     collect(datatype, distance, idx, cnt)
#     cnt += 1
# distance = 100
# for idx in range(1, 5):
#     collect(datatype, distance, idx, cnt)
#     cnt += 1

# datatype = 'mid'
# distance = 110
# for idx in range(1, 5):
#     collect(datatype, distance, idx, cnt)
#     cnt += 1
# distance = 120
# for idx in range(1, 5):
#     collect(datatype, distance, idx, cnt)
#     cnt += 1

# datatype = 'big'
# distance = 120
# for idx in [1, 2]:
#     collect(datatype, distance, idx, cnt)
#     cnt += 1

datatype = 'mid'
for distance in [90, 100, 130]:
    for idx in range(1, 5):
        collect(datatype, distance, idx, cnt)
        cnt += 1

datatype = 'spo'
for distance in [90, 100, 110, 120, 130]:
    idx = 1
    collect(datatype, distance, idx, cnt)
    cnt += 1
