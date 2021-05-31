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
from matting.models.model import theModel
from matting.inference import inference_rotation_multiscale, inference_aug

def get_args():
    parser = argparse.ArgumentParser(description='DeepImageMatting')
    parser.add_argument('--resume', type=str, required=True, help="checkpoint that model resume from")
    parser.add_argument('--testResDir', type=str, required=True, help="where prediction result save to")
    args = parser.parse_args()
    print(args)
    return args

def gen_dataset(imgdir, trimapdir):
        sample_set = []
        img_ids = os.listdir(imgdir)
        img_ids.sort()
        cnt = len(img_ids)
        cur = 1
        for img_id in img_ids:
            img_name = os.path.join(imgdir, img_id)
            trimap_name = os.path.join(trimapdir, img_id)

            assert(os.path.exists(img_name))
            assert(os.path.exists(trimap_name))
            sample_set.append((img_name, trimap_name))

        return sample_set

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

def test(args, model, logger, saveImg = False):
    model.eval()
    sample_set = []
    img_ids = os.listdir(config.test_img_path)
    img_ids = [img_id for img_id in img_ids if img_id[-4:] == ".png"]
    img_ids.sort()
    cnt = len(img_ids)

    unique_ids = {}
    for img_id in img_ids:
        tmp = img_id.split("_")
        del tmp[-1]
        tmp = "_".join(tmp)
        if tmp not in unique_ids.keys():
            unique_ids[tmp] = 1
        else:
            unique_ids[tmp] += 1
    unique_ids_mse = unique_ids.copy()
    for img_id in unique_ids_mse:
        unique_ids_mse[img_id] = 0.
    unique_ids_sad = unique_ids_mse.copy()

    mse_diffs = 0.
    sad_diffs = 0.
    for img_id in img_ids:
        img_path = os.path.join(config.test_img_path, img_id)
        trimap_path = os.path.join(config.test_trimap_path, img_id)

        assert(os.path.exists(img_path))
        assert(os.path.exists(trimap_path))

        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path, 0)

        assert(img.shape[:2] == trimap.shape[:2])

        img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])

        with torch.no_grad():
            torch.cuda.empty_cache()
            origin_pred_mattes = inference_aug(model, img, trimap)

        # only care about the unknown region
        origin_pred_mattes[trimap == 255] = 1.
        origin_pred_mattes[trimap == 0  ] = 0.

        origin_pred_mattes = np.clip(origin_pred_mattes, 0, 1)

        # origin trimap
        pixel = float((trimap == 128).sum())

        # eval if gt alpha is given
        if config.test_alpha_path != '':
            alpha_name = os.path.join(config.test_alpha_path, img_info[0])
            assert(os.path.exists(alpha_name))
            alpha = cv2.imread(alpha_name, 0) / 255.
            assert(alpha.shape == origin_pred_mattes.shape)

            mse_diff = ((origin_pred_mattes - alpha) ** 2).sum() / pixel
            sad_diff = np.abs(origin_pred_mattes - alpha).sum()


            tmp = img_id.split("_")
            del tmp[-1]
            tmp = "_".join(tmp)
            unique_ids_mse[tmp] += mse_diff
            unique_ids_sad[tmp] += sad_diff
            mse_diffs += mse_diff
            sad_diffs += sad_diff

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)

        if saveImg:
            if not os.path.exists(args.testResDir):
                os.makedirs(args.testResDir)
            cv2.imwrite(os.path.join(args.testResDir, img_info[0]), origin_pred_mattes)

    if config.test_alpha_path != '':
        for ids in unique_ids:
            unique_ids_mse[ids] /= unique_ids[ids]
            unique_ids_sad[ids] /= unique_ids[ids]
            #logger.info(" {}: Eval-MSE: {} Eval-SAD: {}".format(ids, unique_ids_mse[ids], unique_ids_sad[ids]))
        logger.info("Eval-MSE: {}".format(mse_diffs / cnt))
        logger.info("Eval-SAD: {}".format(sad_diffs / cnt))
    return sad_diffs / cnt

if __name__ == "__main__":

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    if config.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")

    if not os.path.exists(args.testResDir):
        os.makedirs(args.testResDir)

    logger_test = get_logger(os.path.join(config.saveDir, "log_test.txt"), "testLogger")
    model = theModel()

    model = nn.DataParallel(model)

    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    if config.cuda:
        model = model.cuda()

    test(args, model, logger_test, saveImg = True)
