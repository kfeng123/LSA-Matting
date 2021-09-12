import torch
import argparse
import torch.nn as nn
import cv2
import os
import sys
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import time

from .utils import config
from .models.model import theModel

# inference an img with no tricks
def inference(model, img, trimap):
    h, w = img.shape[0], img.shape[1]
    origin_trimap = trimap.copy()

    # pad to meet h % 64 == 0 and w % 64 == 0
    theDivider = 64
    if h % theDivider == 0:
        pad_h = 0
    else:
        pad_h = theDivider - h % theDivider

    if w % theDivider == 0:
        pad_w = 0
    else:
        pad_w = theDivider - w % theDivider
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), mode = "reflect")
    trimap = np.pad(trimap, ((0, pad_h), (0, pad_w)), mode = "reflect")

    # additional zero padding
    add_pad = 32
    tmp = int(add_pad/2)
    img = np.pad(img, ((tmp, tmp), (tmp, tmp), (0,0)), mode = "reflect")
    img = np.pad(img, ((tmp, tmp), (tmp, tmp), (0,0)), mode = "constant")
    trimap = np.pad(trimap, ((tmp, tmp), (tmp, tmp)), mode = "reflect")
    trimap = np.pad(trimap, ((tmp, tmp), (tmp, tmp)), mode = "constant")

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = config.mean, std = config.std)
    ])

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = normalize(img_rgb).unsqueeze(0)
    tensor_trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])

    if config.cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()

    if config.input_format == "RGB":
        input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)
    elif config.input_format == "BGR":
        input_t = torch.cat((tensor_img[:,[2,1,0],:,:], tensor_trimap / 255.), 1)

    out = model(input_t)
    pred_mattes = out['alpha']
    if config.cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    pred_mattes = np.clip(pred_mattes, 0, 1)

    pred_mattes = pred_mattes[add_pad:(h+add_pad), add_pad:(w+add_pad)]

    pred_mattes[origin_trimap > 200] = 1
    pred_mattes[origin_trimap < 50] = 0
    return pred_mattes

# inference an image with possible augmentations
def inference_aug(model, img, trimap):
    h, w, c = img.shape
    # resize
    if config.max_size < h or config.max_size < w:
        new_h = min(config.max_size, h)
        new_w = min(config.max_size, w)
        # resize for network input, to Tensor
        scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ifResize = True
    else:
        new_h = h
        new_w = w
        scale_img = img.copy()
        scale_trimap = trimap.copy()
        ifResize = False

    pred_mattes = inference(model, scale_img, scale_trimap)

    if ifResize:
        # resize to original size
        pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(pred_mattes.shape == trimap.shape)

    return pred_mattes

# inference an image with rotation
def inference_rotation(model, img, trimap):
    h, w, c = img.shape
    # resize
    if config.max_size < h or config.max_size < w:
        new_h = min(config.max_size, h)
        new_w = min(config.max_size, w)
        # resize for network input, to Tensor
        scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        ifResize = True
    else:
        new_h = h
        new_w = w
        scale_img = img.copy()
        scale_trimap = trimap.copy()
        ifResize = False

    all_preds = []
    # original
    pred_mattes = inference(model, scale_img, scale_trimap)
    all_preds.append(pred_mattes)

    # horizontal flip
    tmp_img = cv2.flip(scale_img, 1)
    tmp_trimap = cv2.flip(scale_trimap, 1)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, 1)
    all_preds.append(pred_mattes)

    # vertical flip
    tmp_img = cv2.flip(scale_img, 0)
    tmp_trimap = cv2.flip(scale_trimap, 0)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, 0)
    all_preds.append(pred_mattes)

    # horizontal and vertical flip
    tmp_img = cv2.flip(scale_img, -1)
    tmp_trimap = cv2.flip(scale_trimap, -1)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, -1)
    all_preds.append(pred_mattes)

    # transpose
    tmp_img = scale_img.transpose((1, 0, 2))
    tmp_trimap = scale_trimap.transpose((1, 0))
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = pred_mattes.transpose((1, 0))
    all_preds.append(pred_mattes)

    # transpose -> horizontal flip
    tmp_img = scale_img.transpose((1, 0, 2))
    tmp_img = cv2.flip(tmp_img, 1)
    tmp_trimap = scale_trimap.transpose((1, 0))
    tmp_trimap = cv2.flip(tmp_trimap, 1)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, 1)
    pred_mattes = pred_mattes.transpose((1, 0))
    all_preds.append(pred_mattes)

    # transpose -> vertical flip
    tmp_img = scale_img.transpose((1, 0, 2))
    tmp_img = cv2.flip(tmp_img, 0)
    tmp_trimap = scale_trimap.transpose((1, 0))
    tmp_trimap = cv2.flip(tmp_trimap, 0)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, 0)
    pred_mattes = pred_mattes.transpose((1, 0))
    all_preds.append(pred_mattes)

    # transpose -> horizontal and vertical flip
    tmp_img = scale_img.transpose((1, 0, 2))
    tmp_img = cv2.flip(tmp_img, -1)
    tmp_trimap = scale_trimap.transpose((1, 0))
    tmp_trimap = cv2.flip(tmp_trimap, -1)
    pred_mattes = inference(model, tmp_img, tmp_trimap)
    pred_mattes = cv2.flip(pred_mattes, -1)
    pred_mattes = pred_mattes.transpose((1, 0))
    all_preds.append(pred_mattes)

    final_matte = np.zeros(all_preds[0].shape)
    for matte in all_preds:
        final_matte = final_matte + matte
    pred_mattes = final_matte / 8

    if ifResize:
        # resize to original size
        pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(pred_mattes.shape == trimap.shape)

    return pred_mattes

def inference_rotation_permute(model, img, trimap):
    h, w, c = img.shape
    all_preds = []

    tmp = inference_rotation(model, scale_img, scale_trimap)
    all_preds.append(tmp)

    tmp = inference_rotation(model, 255 - scale_img, scale_trimap)
    all_preds.append(tmp)

    tmp = inference_rotation(model, scale_img[:,:,[2,1,0]], scale_trimap)
    all_preds.append(tmp)

    final_matte = np.zeros(all_preds[0].shape)
    for matte in all_preds:
        final_matte = final_matte + matte
    pred_mattes = final_matte / 3
    return pred_mattes




def inference_rotation_multiscale(model, img, trimap):
    h, w, c = img.shape
    all_preds = []
    for scale in [0.8, 1, 1.25]:
        new_w = int(scale * w)
        new_h = int(scale * h)
        scale_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation = cv2.INTER_NEAREST)
        tmp = inference_rotation(model, scale_img, scale_trimap)
        tmp = cv2.resize(tmp, (w, h), interpolation = cv2.INTER_CUBIC)
        all_preds.append(tmp)

    final_matte = np.zeros(all_preds[0].shape)
    for matte in all_preds:
        final_matte = final_matte + matte
    pred_mattes = final_matte / 3
    return pred_mattes
