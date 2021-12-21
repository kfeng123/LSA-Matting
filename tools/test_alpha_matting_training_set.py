import torch
import argparse
import torch.nn as nn
import cv2
import os
import sys
sys.path.insert(0,".")
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import time

import matting.utils.config as config
from matting.utils.utils import get_logger
from matting.models.model import theModel, test_time_model
from matting.inference import inference_rotation_multiscale, inference_aug

def get_args():
    parser = argparse.ArgumentParser(description='DeepImageMatting')
    parser.add_argument('--resume', type=str, required=True, help="checkpoint that model resume from")
    args = parser.parse_args()
    print(args)
    return args

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

def test(model, test_img_path, test_trimap_path, test_alpha_path, output_path):
    model.eval()
    sample_set = []
    img_ids = os.listdir(test_img_path)
    img_ids = [img_id for img_id in img_ids if img_id[-4:] == ".png"]
    img_ids.sort()
    cnt = len(img_ids)

    mse_diffs = 0.
    sad_diffs = 0.
    grad_diffs = 0.
    connect_diffs = 0.

    for img_id in img_ids:
        img_path = os.path.join(test_img_path, img_id)
        trimap_path = os.path.join(test_trimap_path, img_id)
        alpha_path = os.path.join(test_alpha_path, img_id)
        assert(os.path.exists(img_path))
        assert(os.path.exists(trimap_path))
        assert(os.path.exists(alpha_path))
        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path, 0)
        assert(img.shape[:2] == trimap.shape[:2])
        img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])
        with torch.no_grad():
            torch.cuda.empty_cache()
            #origin_pred_mattes = inference_rotation_multiscale(model, img, trimap)
            origin_pred_mattes = inference_aug(model, img, trimap)

        origin_pred_mattes[trimap == 255] = 1.
        origin_pred_mattes[trimap == 0  ] = 0.

        origin_pred_mattes = np.clip(origin_pred_mattes, 0, 1)

        alpha = cv2.imread(alpha_path, 0) / 255.
        assert(alpha.shape == origin_pred_mattes.shape)
        mse_diff = eval_mse(origin_pred_mattes, alpha, trimap)
        sad_diff = eval_sad(origin_pred_mattes, alpha, trimap)
        grad_diff = eval_gradient_loss(origin_pred_mattes, alpha, trimap)
        connect_diff = eval_connectivity_loss(origin_pred_mattes, alpha, trimap)

        mse_diffs += mse_diff;
        sad_diffs += sad_diff;
        grad_diffs += grad_diff;
        connenct_diffs += connenct_diff;

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, img_info[0]), origin_pred_mattes)
    print("MSE", mse_diffs / len(img_ids))
    print("SAD", sad_diffs / len(img_ids))
    print("GRAD", grad_diffs / len(img_ids))
    print("CONN", connect_diffs / len(img_ids))
    return 0

if __name__ == "__main__":

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    if config.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")

    model = theModel()
    ###################################
    model = test_time_model(model)
    ###################################
    model = nn.DataParallel(model)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    if config.cuda:
        model = model.cuda()

    test(model,
         "/sda/WangRui/alphaMatting_train/input_training_lowres",
         "/sda/WangRui/alphaMatting_train/trimap_training_lowres/Trimap1",
         "/sda/WangRui/alphaMatting_train/gt_training_lowres",
         "/sda/WangRui/alphaMatting_train/result_training_lowres/Trimap1")
    test(model,
         "/sda/WangRui/alphaMatting_train/input_training_lowres",
         "/sda/WangRui/alphaMatting_train/trimap_training_lowres/Trimap2",
         "/sda/WangRui/alphaMatting_train/gt_training_lowres",
         "/sda/WangRui/alphaMatting_train/result_training_lowres/Trimap2")
