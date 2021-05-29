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
from tensorboardX import SummaryWriter

from matting.data.data_online_official import MatDataset
from matting.models.model import theModel
#from matting.models.model_dilate import theModel
import matting.utils.config as config
from matting.utils.loss import *
from matting.utils.utils import get_logger

from tools.test import test

writer = SummaryWriter("runs")

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Good Image Matting')
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate.')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    args = parser.parse_args()
    return args

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataset(args):
    train_set = MatDataset(args)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=config.threads,
                              batch_size=args.batchSize,
                              shuffle=True,
                              worker_init_fn = my_worker_init_fn)
    return train_loader

def build_model(args, logger):

    model = theModel()
    model = nn.DataParallel(model)

    start_epoch = 1
    best_sad = 100000000.
    if args.pretrain and os.path.isfile(args.pretrain):
        logger.info("loading pretrain '{}'".format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.module.load_state_dict(ckpt['state_dict'],strict=False)
        #logger.info("loaded pretrain '{}' (epoch {})".format(args.pretrain, ckpt['epoch']))

    ifResume = False
    if args.resume and os.path.isfile(args.resume):
        ifResume = True
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']
        best_sad = ckpt['best_sad']
        model.load_state_dict(ckpt['state_dict'],strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {} bestSAD {:.3f})".format(args.resume, ckpt['epoch'], ckpt['best_sad']))

    return start_epoch, model, best_sad, ifResume

def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
    return ss

def train(args, model, optimizer, train_loader, lr_scheduler, epoch, logger):
    model.train()
    t0 = time.time()
    sum_loss = 0
    sum_loss_alpha = 0
    sum_loss_comp = 0
    for iteration, batch in enumerate(train_loader, 1):
        torch.cuda.empty_cache()

        fg_norm = batch[0]
        bg_norm = batch[1]
        alpha = batch[2]
        trimap = batch[3]
        img_info = batch[-1]

        if config.aux_loss_Urysohn:
            optimal_trimap_Urysohn = batch[-2]

        if config.cuda:
            fg_norm = fg_norm.cuda()
            bg_norm = bg_norm.cuda()
            alpha = alpha.cuda()
            trimap = trimap.cuda()
            if config.aux_loss_Urysohn:
                optimal_trimap_Urysohn = optimal_trimap_Urysohn.cuda()


        #if config.input_format == "RGB":
        #    out = model(torch.cat((img_norm, trimap / 255.), 1))
        #if config.input_format == "BGR":
        #    out = model(torch.cat((img_norm[:, [2,1,0],:,:], trimap / 255.), 1))
        img_norm = fg_norm * alpha/255. + bg_norm * (1- alpha/255.)
        if config.input_format == "BGR":
            img_norm = img_norm[:, [2,1,0],:,:]
        out = model(torch.cat((img_norm, trimap / 255.), 1))
        pred_mattes = out['alpha']
        loss = my_alpha_loss_multiscale(alpha, trimap, pred_mattes)

        if config.aux_loss:
            aux_mattes = out['aux_alpha']
            #aux_loss = my_alpha_loss(aux_mattes, trimap, pred_mattes)
            aux_loss = cross_entropy_loss(aux_mattes, pred_mattes, trimap)

        if config.aux_loss_Urysohn:
            aux_Urysohn = out['aux_Urysohn']
            tmp = aux_Urysohn - optimal_trimap_Urysohn
            Urysohn_loss = torch.sqrt( tmp ** 2 + 1e-4 ).sum() / np.prod(tmp.shape)

        #loss = my_alpha_loss_weight(alpha, trimap, pred_mattes, 1e3 * epoch / args.nEpochs )
        #loss = my_alpha_loss_weight_inverse(alpha, trimap, pred_mattes, 10. * (1-epoch / args.nEpochs ))

        sum_loss += loss.item()
        #sum_loss_alpha += alpha_loss.item()
        #sum_loss_comp += comp_loss.item()

        if config.aux_loss:
            total_loss = loss + aux_loss / 5
        elif config.aux_loss_Urysohn:
            total_loss = loss + Urysohn_loss / 5
        else:
            total_loss = loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        #if iteration % 2000 ==0:
        #    for theName, theP in model.named_parameters():
        #        if theP.grad is not None:
        #            logger.info(theName)
        #            logger.info("norm: {:20.4f} max_norm: {:20.4f} grad_l2_norm: {:20.4f} grad_max_norm: {:20.4f}".format(theP.data.norm(2).item(), theP.data.abs().max().item(), theP.grad.data.norm(2).item(), theP.grad.data.abs().max().item()))

        #nn.utils.clip_grad_norm_(model.parameters(), 5)

        if iteration % config.printFreq ==  0:
            t1 = time.time()
            num_iter = len(train_loader)
            speed = (t1 - t0) / iteration
            exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))

            logger.info("Epoch[{}/{}]({}/{}) Lr:{:.8f} Avg_Loss:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], sum_loss/iteration, speed, exp_time))

            #niter = epoch * len(train_loader) + iteration
            #writer.add_scalar("Train/Loss", loss.item(), niter)
            #writer.add_scalar("Train/Loss_alpha", alpha_loss.item(), niter)
            #writer.add_scalar("Train/Loss_comp", comp_loss.item(), niter)

        lr_scheduler.step()

def checkpoint(epoch, save_dir, model, best_sad, logger, best=False):

    epoch_str = "best" if best else "e{}".format(epoch)
    model_out_path = "{}/ckpt_{}.pth".format(save_dir, epoch_str)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_sad': best_sad
    }, model_out_path )
    logger.info("Checkpoint saved to {}".format(model_out_path))

def main():

    args = get_args()
    logger = get_logger(os.path.join(config.saveDir, "log.txt"), "mainLogger")
    logger.info("Loading args: \n{}".format(args))

    logger_test = get_logger(os.path.join(config.saveDir, "log_test.txt"), "testLogger")

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if config.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    if config.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logger.info("Loading dataset:")
    train_loader = get_dataset(args)

    logger.info("Building model:")
    start_epoch, model, best_sad, ifResume = build_model(args, logger)
    if config.cuda:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
        #torch.backends.cudnn.deterministic = True

    backbone_params = []
    prelu_params = []
    norm_params = []
    learning_params = []
    for param in model.named_parameters():
        if "norm" in param[0] or "bn" in param[0]:
            norm_params.append(param[1])
        elif "prelu" in param[0]: #or "bias" in param[0]:
            prelu_params.append(param[1])
        else:
            learning_params.append(param[1])
    param_groups = [
                {"params": learning_params, "initial_lr": args.lr},
                {"params": norm_params, "weight_decay": 1e-5, "initial_lr": args.lr},
                {"params": prelu_params, "weight_decay": 0, "initial_lr": args.lr},
            ]

    if config.opt_method == "Adam":
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay = config.weight_decay)

    def lr_func(iteration):
        tmp = int(iteration / (len(train_loader)) / 20)
        return 1/(2 ** tmp)
        #if iteration / (len(train_loader)) >=160:
        #    return 0.01
        #if iteration / (len(train_loader)) >=120:
        #    return 0.02
        #if iteration / (len(train_loader)) >=80:
        #    return 0.04
        #if iteration / (len(train_loader)) >=40:
        #    return 0.2
        #return 1.

    #lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func, -1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_func, (start_epoch-1)*len(train_loader)-1)

    # training
    for epoch in range(start_epoch, args.nEpochs + 1):
        train(args, model, optimizer, train_loader, lr_scheduler, epoch, logger)
        if epoch > 0 and epoch % config.ckptSaveFreq == 0:
            checkpoint(epoch, config.saveDir, model, best_sad, logger)
        if epoch > 0 and config.testFreq > 0 and epoch % config.testFreq == 0:
            logger_test.info("Epoch: {}".format(epoch))
            if epoch >=5:
                cur_sad = test(args, model, logger_test)
                if cur_sad < best_sad:
                    best_sad = cur_sad
                    checkpoint(epoch, config.saveDir, model, best_sad, logger, True)


if __name__ == "__main__":
    main()
