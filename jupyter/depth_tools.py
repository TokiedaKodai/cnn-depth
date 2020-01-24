import cv2
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

def unpack_png_to_float(png):
    depthImageUnit = 0.00001
    png = png.astype(float)
    depth = (png[:, :, 0] + png[:, :, 1] * 256 +
             png[:, :, 2] * 256 * 256) * depthImageUnit
    return depth


def pack_float_to_bmp_bgra(depth):
    m, e = np.frexp(depth)
    m = (m * (256**3)).astype(np.uint64)
    bmp = np.zeros((*depth.shape[:2], 4), np.uint8)
    bmp[:, :, 0] = (e + 128).astype(np.uint8)
    bmp[:, :, 1] = np.right_shift(np.bitwise_and(m, 0x00ff0000), 16)
    bmp[:, :, 2] = np.right_shift(np.bitwise_and(m, 0x0000ff00), 8)
    bmp[:, :, 3] = np.bitwise_and(m, 0x000000ff)
    return bmp


def unpack_bmp_bgra_to_float(bmp):
    b = bmp[:, :, 0].astype(np.int32)
    g = bmp[:, :, 1].astype(np.int32) << 16
    r = bmp[:, :, 2].astype(np.int32) << 8
    a = bmp[:, :, 3].astype(np.int32)
    depth = np.ldexp(1.0, b -
                     (128 + 24)) * (g + r + a + 0.5).astype(np.float32)
    return depth


def unpack_bmp_bgr_to_float(bmp):
    b = bmp[:, :, 0].astype(np.int32)
    g = bmp[:, :, 1].astype(np.int32) << 16
    r = bmp[:, :, 2].astype(np.int32) << 8
    depth = np.ldexp(1.0, b - (128 + 24)) * (g + r + 0.5).astype(np.float32)
    return depth


def convert_depth_to_coords(raw_depth, cam_params):
    h, w = raw_depth.shape[:2]

    # convert depth to 3d coord
    xs, ys = np.meshgrid(range(w), range(h))

    z = raw_depth
    dist_x = cam_params['pix_x'] * (xs - cam_params['center_x'])
    dist_y = -cam_params['pix_y'] * (ys - cam_params['center_y'])
    xyz = np.vstack([(dist_x * z / cam_params['focal_length']).flatten(),
                     (dist_y * z / cam_params['focal_length']).flatten(),
                     -z.flatten()]).T.reshape((h, w, -1))
    return xyz

def generate_faces(shape, mask=None):
    mask = np.ones(shape) if mask is None else mask
    # re-index for only valid pixels
    h, w = mask.shape[:2]
    v_indices = np.zeros((h * w))
    valid_indices = np.where(mask.flatten())[0]
    v_indices[valid_indices] = np.arange(np.count_nonzero(mask))
    v_indices = v_indices.reshape((h, w))
    '''
    mesh for first-case
    v0-v2
    | /
    v1
    '''
    v0 = v_indices[:h - 1, :w - 1]
    v1 = v_indices[1:, :w - 1]
    v2 = v_indices[:h - 1, 1:]
    faces_0 = np.dstack([v0, v1, v2]).reshape((-1, 3))
    face_mask = np.logical_and(
        np.logical_and(mask[:h - 1, :w - 1], mask[1:, :w - 1]),
        mask[:h - 1, 1:]).flatten()
    faces_0 = faces_0[np.where(face_mask.flatten())[0]]
    '''
    mesh for second-case
        v0
        / |
    v1-v2
    '''
    v0 = v_indices[:h - 1, 1:]
    v1 = v_indices[1:, :w - 1]
    v2 = v_indices[1:, 1:]
    faces_1 = np.dstack([v0, v1, v2]).reshape((-1, 3))
    face_mask = np.logical_and(
        np.logical_and(mask[:h - 1, 1:], mask[1:, :w - 1]),
        mask[1:, 1:]).flatten()
    faces_1 = faces_1[np.where(face_mask)[0]]

    return np.r_[faces_0, faces_1]


def dump_ply(filename, points, colors=None, faces=None):
    params = []

    minimum = 0.00001

    # params.append(len(points))
    arr_points = np.array(points)
    length = np.sum(np.where(np.abs(arr_points[:, 2]) > minimum, 1, 0))
    params.append(length)

    header = 'ply\n'
    header += 'format ascii 1.0\n'
    header += 'element vertex {:d}\n'
    header += 'property float x\n'
    header += 'property float y\n'
    header += 'property float z\n'
    header += 'property uchar red\n'
    header += 'property uchar green\n'
    header += 'property uchar blue\n'

    if faces is not None:
        params.append(len(faces))
        header += 'element face {:d}\n'
        header += 'property list uchar int vertex_indices \n'

    header += 'end_header\n'

    colors = colors if colors is not None else [[255, 255, 255]] * len(points)
    with open(filename, 'w') as f:
        f.write(header.format(*params))
        for p, color in zip(points, colors):
            # data = (p[0], p[1], p[2], color[0], color[1], color[2])
            # f.write('%f %f %f %u %u % u\n' % data)
            if np.abs(p[2]) > minimum:
                data = (p[0], p[1], p[2], color[0], color[1], color[2])
                f.write('%f %f %f %u %u %u\n' % data)

        faces = faces if faces is not None else []
        for v in faces:
            data = (v[0], v[1], v[2])
            f.write('3 %d %d %d \n' % data)


def read_depth_test():
    srcs = ['check13-depthimage.png']

    for src in srcs:
        png = cv2.imread(src, -1)
        depth = unpack_png_to_float(png)

        bmp = pack_float_to_bmp_bgra(depth)

        depth_check = unpack_bmp_bgra_to_float(bmp)

        plt.imshow(depth)
        plt.colorbar()

        plt.figure()
        plt.imshow(bmp)

        plt.figure()
        plt.imshow(depth_check)
        plt.colorbar()

        plt.figure()
        plt.hist(depth[depth > 0].flatten(), range=[0, 1.5])
        plt.show()


def compare_gt_amida():
    # src_amida = 'check13-depthimage.png'
    # src_gt = 'NewScan000.depth.dist.bmp'
    src_amida = './input_data_0725/rec/depth00000.png'
    src_gt = './input_data_0725/gt/gt00000.bmp'

    # unpack depth from image
    depth_amida = unpack_png_to_float(cv2.imread(src_amida, -1))
    depth_gt = unpack_bmp_bgra_to_float(cv2.imread(src_gt, -1))

    # plt.figure()
    # plt.imshow(depth_gt)
    # # plt.imshow(depth_gt, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.show()

    # # convert depth to 3d points (amida)
    # cam_params = {
    #     'focal_length': 0.056618,
    #     'pix_x': 0.0000195313,
    #     'pix_y': 0.0000195620,
    #     'center_x': 800,
    #     'center_y': 600
    # }
    cam_params = {
        'focal_length': 0.057615,
        'pix_x': 0.0000195313,
        'pix_y': 0.0000195539,
        'center_x': 600,
        'center_y': 600
    }
    xyz_amida = convert_depth_to_coords(depth_amida, cam_params)
    # texture = cv2.imread('texture.bmp', -1)[:, 0:1200].reshape(
    #     (-1, 3)).tolist()

    # convert depth to 3d points (gt)
    # cam_params = {
    #     'focal_length': 0.056618,
    #     'pix_x': 0.0000195313,
    #     'pix_y': 0.0000195620,
    #     'center_x': 800,
    #     'center_y': 600
    # }
    cam_params = {
        'focal_length': 0.057615,
        'pix_x': 0.0000195313,
        'pix_y': 0.0000195539,
        'center_x': 600,
        'center_y': 600
    }

    xyz_gt = convert_depth_to_coords(depth_gt, cam_params)
    # texture = cv2.imread('texture.bmp', -1).reshape((-1, 3)).tolist()

    # save 3d points to ply file
    dump_ply('test-amida.ply', xyz_amida.reshape(-1, 3).tolist())
    dump_ply('test-gt.ply', xyz_gt.reshape(-1, 3).tolist())

    # compare depth image
    vmin, vmax = 0.4, 1.2
    plt.figure()
    # plt.imshow(depth_amida)
    plt.imshow(depth_amida, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.figure()
    plt.imshow(depth_gt)
    plt.imshow(depth_gt, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # read_depth_test()
    compare_gt_amida()
