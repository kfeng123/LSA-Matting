import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os
import sys
sys.path.insert(0,".")
import cv2
import numpy as np
import logging
import random

import matting.utils.config as config
from matting.data.data_no_aug import MatDataset
from matting.models.model import theModel
from matting.utils.loss import *
from matting.utils.utils import get_logger
from tools.test import test

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Fine Image Matting')
    parser.add_argument('--resume', type=str, help="Checkpoint that the model resume from")
    args = parser.parse_args()
    return args

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataset(args):
    train_set = MatDataset(args)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=config.threads,
                              batch_size=1,
                              shuffle=True,
                              worker_init_fn = my_worker_init_fn)
    return train_loader

def build_model(args, logger):

    model = theModel()
    model = nn.DataParallel(model)

    start_epoch = 1

    ifResume = False
    if args.resume and os.path.isfile(args.resume):
        ifResume = True
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'],strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    return start_epoch, model, ifResume

def train(args, model, optimizer, train_loader, lr_scheduler, epoch, logger):
    model.train()
    t0 = time.time()
    sum_loss = 0
    sum_loss_alpha = 0
    sum_loss_comp = 0
    for iteration, batch in enumerate(train_loader, 1):
        torch.cuda.empty_cache()

        img_norm = batch[0]
        alpha = batch[1]
        trimap = batch[2]
        img_info = batch[-1]

        if config.cuda:
            img_norm = img_norm.cuda()
            alpha = alpha.cuda()
            trimap = trimap.cuda()

        if config.input_format == "BGR":
            img_norm = img_norm[:, [2,1,0],:,:]
        out = model(torch.cat((img_norm, trimap / 255.), 1))
        pred_mattes = out['alpha']
        loss = my_alpha_loss_multiscale(alpha, trimap, pred_mattes)
        #loss_coarse = my_alpha_loss_multiscale(alpha, trimap, out['alpha_coarse'])
        sum_loss += loss.item()
        total_loss = loss #+ 0.2 * loss_coarse
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iteration % config.printFreq ==  0:
            t1 = time.time()
            num_iter = len(train_loader)
            speed = (t1 - t0) / iteration

            logger.info("Epoch[{}/{}]({}/{}) Lr:{:.7f} Avg_Loss:{:.3f} Speed:{:.2f}s/iter".format(epoch, config.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], sum_loss/iteration, speed))

        lr_scheduler.step()

def checkpoint(epoch, save_dir, model, logger):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_str = "e{}".format(epoch)
    model_out_path = "{}/ckpt_{}.pth".format(save_dir, epoch_str)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, model_out_path )
    logger.info("Checkpoint saved to {}".format(model_out_path))

def main():

    args = get_args()
    logger = get_logger(os.path.join(config.saveDir, "log.txt"), "mainLogger")
    logger.info("Loading args: \n{}".format(args))

    logger_test = get_logger(os.path.join(config.saveDir, "log_test.txt"), "testLogger")

    if config.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    if config.cuda:
        torch.cuda.manual_seed(111)
    else:
        torch.manual_seed(111)

    logger.info("Loading dataset:")
    train_loader = get_dataset(args)

    logger.info("Building model:")
    start_epoch, model, ifResume = build_model(args, logger)
    if config.cuda:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    backbone_params = []
    prelu_params = []
    norm_params = []
    learning_params = []
    for param in model.named_parameters():
        if "norm" in param[0] or "bn" in param[0]:
            norm_params.append(param[1])
        elif "prelu" in param[0]:
            prelu_params.append(param[1])
        else:
            learning_params.append(param[1])
    param_groups = [
                {"params": learning_params, "initial_lr": config.lr},
                {"params": norm_params, "weight_decay": 1e-5, "initial_lr": config.lr},
                {"params": prelu_params, "weight_decay": 0, "initial_lr": config.lr},
            ]

    if config.opt_method == "Adam":
        optimizer = optim.Adam(param_groups, lr=config.lr, weight_decay = config.weight_decay)

    def lr_func(iteration):
        tmp = int(iteration / (len(train_loader)) / 20)
        return 1/(2 ** tmp)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func, (start_epoch-1)*len(train_loader)-1)

    # training
    for epoch in range(start_epoch, config.nEpochs + 1):
        train(args, model, optimizer, train_loader, lr_scheduler, epoch, logger)
        if epoch > 0 and epoch % config.ckptSaveFreq == 0:
            checkpoint(epoch, config.saveDir, model, logger)
        if epoch > 0 and config.testFreq > 0 and epoch % config.testFreq == 0:
            logger_test.info("Epoch: {}".format(epoch))
            test(args, model, logger_test)

if __name__ == "__main__":
    main()
