# This code is a python version of the matlab code provided by the website of ``Deep Image Matting'' (https://sites.google.com/view/deepimagematting)

import numpy as np
import math
from scipy import ndimage
from skimage import measure
# both pred and gt are numpy float arrays, dim = (height, width), taking values in the interval [0,1]
# timap is a numpy uint8 array, dim = (height, width). 255 is foreground, 128 is unknown, 0 is background

def eval_mse(pred, gt, trimap):
    pixel = float((trimap == 128).sum())
    return ((pred - gt) ** 2).sum() / pixel

def eval_sad(pred, gt, trimap):
    return np.abs(pred - gt).sum() / 1000.

def gauss(x, sigma):
    y = math.exp(-x**2 / (2* sigma ** 2 )) / (sigma* math.sqrt(2*math.pi))
    return y

def dgauss(x, sigma):
    y = - x * gauss(x, sigma) / sigma ** 2
    return y


def gaussgradient(img, sigma):
    epsilon = 1e-2
    halfsize = sigma * math.sqrt(- 2 * math.log(math.sqrt(2*math.pi) * sigma * epsilon ))
    halfsize = int(halfsize)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            index1 = i - halfsize
            index2 = j - halfsize
            hx[i,j] = gauss(index1, sigma) * dgauss(index2, sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) ** 2 ))
    hy = hx.T
    gx = ndimage.convolve(img, hx, mode = 'nearest' )
    gy = ndimage.convolve(img, hy, mode = 'nearest' )
    return gx, gy

def eval_gradient_loss(pred, gt, trimap):
    pred_x , pred_y =gaussgradient(pred , 1.4 )
    pred_map = np.sqrt(pred_x ** 2 + pred_y ** 2)
    gt_x , gt_y =gaussgradient(gt , 1.4 )
    gt_map = np.sqrt(gt_x ** 2 + gt_y ** 2)

    error_map = ( pred_map - gt_map ) ** 2
    return np.sum(error_map * (trimap == 128)) / 1000.


def eval_connectivity_loss(pred, gt, trimap):
    h, w = pred.shape
    step = 0.1
    thresh_steps = np.arange(0,1+step, step)
    l_map = np.zeros(pred.shape) - 1
    dist_maps = np.zeros((h, w, thresh_steps.size))
    for ii in range(1,thresh_steps.size):
        pred_alpha_thresh = pred >= thresh_steps[ii]
        gt_alpha_thresh = gt >= thresh_steps[ii]

        tmp_map = (pred_alpha_thresh * gt_alpha_thresh) * 1
        if np.sum(tmp_map) == 0:
            continue
        cc = measure.label(
            tmp_map,
            background = 0,
            connectivity = 1
        )
        cc = measure.regionprops(cc)
        size_vec = [c.area for c in cc]
        max_id = np.argmax(size_vec)
        coords =cc[max_id].coords
        omega = np.zeros((h,w))
        omega[coords[:,0], coords[:,1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag] = thresh_steps[ii - 1]
    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    gt_d = gt - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15)
    gt_phi = 1 - gt_d * (gt_d >= 0.15)
    loss = np.sum(np.abs(pred_phi - gt_phi) * (trimap == 128))
    return loss / 1000.
