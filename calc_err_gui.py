import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

import depth_tools

# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

DIR = '../data/input_200318/'

argv = sys.argv
_, out_dir, epoch_num, idx = argv
idx = int(idx)

# out_dir = '../output/output_' + out_dir
out_dir = '../output/archive/200318/output_' + out_dir

predict_dir = out_dir + '/predict_{}'.format(epoch_num)
predict_dir = out_dir + '/predict_{}_test'.format(epoch_num)

img_h, img_w = 1200, 1200
space_top, space_center = 100, 20
window_h, window_w = img_h + space_top, img_w*2 + space_center
L_h_start, L_h_end = space_top, window_h
L_w_start, L_w_end = 0, img_w
R_h_start, R_h_end = space_top, window_h
R_w_start, R_w_end = img_w + space_center, img_w*2 + space_center
select_bar = 5

drawing = False
mode = True
ix, iy = -1, -1
select_L = False

color_rectangle = (0, 255, 0)
color_eraser = (0, 0, 255)

def circle(img, x, y):
    cv2.circle(img, (x,y), 1, (0,0,255), -1)
    print(x, y)

def rectangle(img, ix, iy, x, y):
    global select_L

    cv2.rectangle(img, (ix, iy), (x,y), color_rectangle, -1)

    if select_L:
        ix -= L_w_start
        x -= L_w_start
        iy -= L_h_start
        y -= L_h_start

        cv2.rectangle(img, (R_w_start + ix, R_h_start + iy), (R_w_start + x, R_h_start + y), color_rectangle, -1)
    else:
        ix -= R_w_start
        x -= R_w_start
        iy -= R_h_start
        y -= R_h_start

        cv2.rectangle(img, (L_w_start + ix, L_h_start + iy), (L_w_start + x, L_h_start + y), color_rectangle, -1)
        
    if x > ix:
        if y > iy:
            mask[iy:y, ix:x] = 1
        else:
            mask[y:iy, ix:x] = 1
    else:
        if y > iy:
            mask[iy:y, x:ix] = 1
        else:
            mask[y:iy, x:ix] = 1
    # print(x, y)

def eraser(img, ix, iy, x, y):
    global select_L, window

    cv2.rectangle(img, (ix, iy), (x,y), color_eraser, -1)

    if x > ix:
        if y > iy:
            window[iy:y, ix:x] = original_window[iy:y, ix:x]
        else:
            window[y:iy, ix:x] = original_window[y:iy, ix:x]
    else:
        if y > iy:
            window[iy:y, x:ix] = original_window[iy:y, x:ix]
        else:
            window[y:iy, x:ix] = original_window[y:iy, x:ix]

    if select_L:
        ix -= L_w_start
        x -= L_w_start
        iy -= L_h_start
        y -= L_h_start

        cv2.rectangle(img, (R_w_start + ix, R_h_start + iy), (R_w_start + x, R_h_start + y), color_eraser, -1)
    else:
        ix -= R_w_start
        x -= R_w_start
        iy -= R_h_start
        y -= R_h_start

        cv2.rectangle(img, (L_w_start + ix, L_h_start + iy), (L_w_start + x, L_h_start + y), color_eraser, -1)
        
    if x > ix:
        if y > iy:
            window[iy:y, ix:x] = original_window[iy:y, ix:x]
            mask_eraser[iy:y, ix:x] = 1
        else:
            window[y:iy, ix:x] = original_window[y:iy, ix:x]
            mask_eraser[y:iy, ix:x] = 1
    else:
        if y > iy:
            window[iy:y, x:ix] = original_window[iy:y, x:ix]
            mask_eraser[iy:y, x:ix] = 1
        else:
            window[y:iy, x:ix] = original_window[y:iy, x:ix]
            mask_eraser[y:iy, x:ix] = 1

# def draw_on_window(event, x, y, flags, param):
#     global ix, iy, drawing, mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 circle(window, x, y)
#             else:
#                 rectangle(window, ix, iy, x, y)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             circle(window, x, y)
#         else:
#             rectangle(window, ix, iy, x, y)

def draw_on_window(event, x, y, flags, param):
    global ix, iy, drawing, select_L

    if (select_L and x in range(L_w_start, L_w_end) and y in range(L_h_start, L_h_end)) or \
    (not select_L and x in range(R_w_start, R_w_end) and y in range(R_h_start, R_h_end)):

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    rectangle(window, ix, iy, x, y)
                else:
                    eraser(window, ix, iy, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                rectangle(window, ix, iy, x, y)
            else:
                eraser(window, ix, iy, x, y)

img_gt = cv2.imread(DIR + 'gt/gt{:03d}.bmp'.format(idx), -1)
img_rec = cv2.imread(DIR + 'rec_ajusted/depth{:03d}.bmp'.format(idx), -1)
img_pred = cv2.imread(predict_dir + '/predict_depth-{:03d}.bmp'.format(idx), -1)

depth_gt = depth_tools.unpack_bmp_bgra_to_float(img_gt)
depth_rec = depth_tools.unpack_bmp_bgra_to_float(img_rec)
depth_pred = depth_tools.unpack_bmp_bgra_to_float(img_pred)

err_abs_rec = np.abs(depth_gt - depth_rec)
err_abs_pred = np.abs(depth_gt - depth_pred)

err_sqr_rec = np.square(depth_gt - depth_rec)
err_sqr_pred = np.square(depth_gt - depth_pred)

window = np.zeros((window_h, window_w))
window[L_h_start:L_h_end, L_w_start:L_w_end] = err_abs_rec
window[R_h_start:R_h_end, R_w_start:R_w_end] = err_abs_pred

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('window', draw_on_window)

vmin_e, vmax_e = 0, 0.005

window = np.where(window > vmax_e, 2**16 - 1, (window / vmax_e)*(2**16 - 1))
window = (window / 256).astype(np.uint8)
window = cv2.applyColorMap(window, cv2.COLORMAP_JET)

mask = np.zeros((img_h, img_w), np.uint8)
mask_eraser = np.zeros((img_h, img_w), np.uint8)

original_window = window

while(1):
    cv2.imshow('window', window)
    cv2.rectangle(window, (0, 0), (window_w, space_top), (0, 0, 0), -1)
    cv2.rectangle(window, (L_w_end, L_h_start), (R_w_start, R_h_end), (0, 0, 0), -1)

    if select_L:
        cv2.rectangle(window, (L_w_start, L_h_start - select_bar), (L_w_end, L_h_start), (0, 0, 255), -1)
        cv2.rectangle(window, (R_w_start, R_h_start - select_bar), (R_w_end, R_h_start), (0, 0, 0), -1)
    else:
        cv2.rectangle(window, (L_w_start, L_h_start - select_bar), (L_w_end, L_h_start), (0, 0, 0), -1)
        cv2.rectangle(window, (R_w_start, R_h_start - select_bar), (R_w_end, R_h_start), (0, 0, 255), -1)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == ord('1'):
        select_L = True
    elif k == ord('2'):
        select_L = False
    elif k == 27:
        break
cv2.destroyAllWindows()

mask = np.where(mask_eraser == 1, 0, mask)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask*255)
cv2.waitKey(0)
cv2.destroyAllWindows()


mask_length = np.sum(mask)

if mask_length == 0:
    print('Not selected')
else:
    MSE_rec = np.sum(err_sqr_rec * mask) / mask_length
    MSE_pred = np.sum(err_sqr_pred * mask) / mask_length

    # Root Mean Square Error
    RMSE_rec = np.sqrt(MSE_rec)
    RMSE_pred = np.sqrt(MSE_pred)
    print('RMSE rec : ', RMSE_rec)
    print('RMSE pred: ', RMSE_pred)
