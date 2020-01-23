import cv2
import numpy as np
from tqdm import tqdm
import depth_tools as tool

IN = 'input_200117'
GT = IN + 'gt/'
REC = IN + 'rec/'
OUT = IN + 'ply/'

depth_threshold = 0.2
difference_threshold = 0.01

# input_200117
cam_params = {
    'focal_length': 0.037297750000000005,
    'pix_x': 1.25e-05,
    'pix_y': 1.237130414015908e-05,
    'center_x': 826.396,
    'center_y': 578.887
}

index = range(1)

for i in tqdm(index):
    gt = GT + 'gt%05d.bmp'%i
    rec = REC + 'depth%05d.png'%i
    # gt = GT + 'NewScan000.depth.dist.bmp'
    # rec = REC + 'check13-depthimage.png'

    depth_gt = tool.unpack_bmp_bgra_to_float(cv2.imread(gt, -1))
    depth_rec = tool.unpack_png_to_float(cv2.imread(rec, -1))
    # depth_err = np.abs(depth_gt - depth_rec) + 1

    is_gt_available = depth_gt > depth_threshold
    is_depth_close = np.logical_and(
        np.abs(depth_rec - depth_gt) < difference_threshold,
        is_gt_available)
    mask = is_depth_close * 1.0

    xyz_gt = tool.convert_depth_to_coords(depth_gt, cam_params)
    xyz_rec = tool.convert_depth_to_coords(depth_rec, cam_params)
    xyz_rec_masked = tool.convert_depth_to_coords(depth_rec * mask, cam_params)
    # xyz_err = tool.convert_depth_to_coords(depth_err, cam_params)
    # print(xyz_gt.shape)
    # print(xyz_gt.reshape(-1, 3).tolist().shape)

    tool.dump_ply(OUT + 'gt%05d.ply'%i, xyz_gt.reshape(-1, 3).tolist())
    tool.dump_ply(OUT + 'rec%05d.ply'%i, xyz_rec.reshape(-1, 3).tolist())
    tool.dump_ply(OUT + 'rec_masked%05d.ply'%i, xyz_rec_masked.reshape(-1, 3).tolist())
    # tool.dump_ply(OUT + 'err%05d.ply'%i, xyz_err.reshape(-1, 3).tolist())