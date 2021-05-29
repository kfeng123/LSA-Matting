

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(three, alpha, trimap):
    trimap_opt = torch.zeros(alpha.shape, dtype = torch.long).cuda()
    trimap_opt[alpha> 0] = 1
    trimap_opt[alpha == 255] = 2
    weighted = (trimap == 128).float()
    tmp = F.cross_entropy(three, trimap_opt[:,0,:,:], reduction = "none")

    tmp = (tmp * weighted[:,0,:,:]).sum() / weighted.sum()
    return tmp

def gen_simple_alpha_loss(alpha, trimap, pred_mattes):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)

    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)

    return alpha_loss_weighted

def my_alpha_loss(alpha, trimap, pred_mattes):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.

    alpha_f = alpha / 255.
    pred_mattes_detach = pred_mattes.detach()
    tmp1 = ((alpha_f>0.999) * (pred_mattes_detach>1)).to(torch.float)
    tmp2 = ((alpha_f < 0.001) * (pred_mattes_detach<0)).to(torch.float)
    alpha_f = (tmp1+tmp2) * pred_mattes_detach + (1-tmp1-tmp2) * alpha_f
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-4)
    #alpha_loss = diff.abs()

    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)

    return alpha_loss_weighted

def my_alpha_loss_add_l2(alpha, trimap, pred_mattes):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.

    alpha_f = alpha / 255.
    pred_mattes_detach = pred_mattes.detach()
    tmp1 = ((alpha_f>0.999) * (pred_mattes_detach>1)).to(torch.float)
    tmp2 = ((alpha_f < 0.001) * (pred_mattes_detach<0)).to(torch.float)
    alpha_f = (tmp1+tmp2) * pred_mattes_detach + (1-tmp1-tmp2) * alpha_f
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    diff_squared = diff ** 2
    alpha_loss_l2 = diff_squared.sum()
    alpha_loss_l1 = torch.sqrt( diff_squared + 1e-4).sum()

    alpha_loss_l1_weighted = alpha_loss_l1 / (weighted.sum() + 1.)

    alpha_loss_l2_weighted = alpha_loss_l2 / (2 * alpha_loss_l1.detach() + 1e-4 )

    return alpha_loss_l1_weighted + 0.25 * alpha_loss_l2_weighted

def my_alpha_loss_multiscale(alpha, trimap_big, pred_mattes):
    trimap = trimap_big[:,0,:,:].unsqueeze(1)
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.

    alpha_f = alpha / 255.
    pred_mattes_detach = pred_mattes.detach()
    tmp1 = ((alpha_f>0.999) * (pred_mattes_detach>1)).to(torch.float)
    tmp2 = ((alpha_f < 0.001) * (pred_mattes_detach<0)).to(torch.float)
    alpha_f = (tmp1+tmp2) * pred_mattes_detach + (1-tmp1-tmp2) * alpha_f
    diff = pred_mattes - alpha_f
    #diff = diff * weighted

    loss_sum = torch.tensor(0).cuda()
    for i in range(5):
        alpha_loss = torch.sqrt(diff ** 2 + 1e-6)
        alpha_loss_weighted = (alpha_loss * weighted).sum() / (weighted.sum() + 1.)
        loss_sum = loss_sum + alpha_loss_weighted / 2 ** i
        diff = F.avg_pool2d(diff, kernel_size = 2)
        weighted = F.avg_pool2d(weighted, kernel_size = 2)

    return loss_sum

def my_alpha_loss_weight(alpha, trimap, pred_mattes, theMul):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.

    alpha_f = alpha / 255.
    tmp = (theMul * alpha_f * (1-alpha_f) + 1)
    weighted = weighted * tmp

    pred_mattes_detach = pred_mattes.detach()
    tmp1 = ((alpha_f>0.99) * (pred_mattes_detach>1)).to(torch.float)
    tmp2 = ((alpha_f < 0.01) * (pred_mattes_detach<0)).to(torch.float)
    alpha_f = (tmp1+tmp2) * pred_mattes_detach + (1-tmp1-tmp2) * alpha_f
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = diff ** 2

    alpha_loss_weighted = alpha_loss.sum() / ( (weighted**2).sum() + 1.)

    return alpha_loss_weighted

def my_alpha_loss_weight_inverse(alpha, trimap, pred_mattes, theMul):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.

    alpha_f = alpha / 255.
    tmp = (1/2-alpha_f * (1-alpha_f)) ** theMul
    weighted = weighted * tmp

    pred_mattes_detach = pred_mattes.detach()
    tmp1 = ((alpha_f>0.99) * (pred_mattes_detach>1)).to(torch.float)
    tmp2 = ((alpha_f < 0.01) * (pred_mattes_detach<0)).to(torch.float)
    alpha_f = (tmp1+tmp2) * pred_mattes_detach + (1-tmp1-tmp2) * alpha_f
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = diff ** 2

    alpha_loss_weighted = alpha_loss.sum() / ( (weighted**2).sum() + 1.)

    return alpha_loss_weighted

def gen_loss(img, alpha, fg, bg, trimap, pred_mattes):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi.cuda()
    t3_wi = torch.cat((wi, wi, wi), 1).cuda()
    unknown_region_size = t_wi.sum()

    #assert(t_wi.shape == pred_mattes.shape)
    #assert(t3_wi.shape == img.shape)

    # alpha diff
    alpha = alpha / 255.
    alpha_loss = torch.sqrt((pred_mattes - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)

    # composite rgb loss
    pred_mattes_3 = torch.cat((pred_mattes, pred_mattes, pred_mattes), 1)
    comp = pred_mattes_3 * fg + (1. - pred_mattes_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
    comp_loss = (comp_loss * t3_wi).sum() / (unknown_region_size + 1.) / 3.

    #print("Loss: AlphaLoss:{} CompLoss:{}".format(alpha_loss, comp_loss))
    return alpha_loss, comp_loss


def gen_alpha_pred_loss(alpha, pred_alpha, trimap):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi.cuda()
    unknown_region_size = t_wi.sum()

    # alpha diff
    alpha = alpha / 255.
    alpha_loss = torch.sqrt((pred_alpha - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)

    return alpha_loss
