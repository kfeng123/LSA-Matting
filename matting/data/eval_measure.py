import numpy as np
import cv2
import math
import pdb
import torch
import torch.nn.functional as F
class gaussGradient():
    def __init__(self, sigma):
        epsilon = 1e-2
        halfsize = math.ceil( sigma * math.sqrt(-2 * math.log( math.sqrt(2* math.pi ) * sigma * epsilon )) )
        size = 2 * halfsize + 1
        self.halfsize = halfsize
        hx = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                u = (i - halfsize, j- halfsize)
                hx[i,j] = self.gauss(u[0], sigma) * self.dgauss(u[1], sigma)
        hx /=math.sqrt( (hx**2).sum())
        hy = hx.copy().transpose()
        self.hx = torch.FloatTensor(hx.astype(float)[np.newaxis, np.newaxis, :, :])
        self.hy = torch.FloatTensor(hy.astype(float)[np.newaxis, np.newaxis, :, :])

    def gauss(self, x, sigma):
        return math.e**( - x**2 / (2*sigma**2) ) /(sigma * math.sqrt( 2* math.pi ))

    def dgauss(self, x, sigma):
        return -x * self.gauss(x, sigma) /sigma**2
    def __call__(self, img):
        gx = F.conv2d(img, self.hx, padding = self.halfsize)
        gy = F.conv2d(img, self.hy, padding = self.halfsize)
        return gx, gy


def compute_gradient_loss(pred, target, trimap, theGaussGradient):
    pred_x, pred_y = theGaussGradient(pred)
    target_x, target_y = theGaussGradient(target)
    pred_amp = (pred_x**2 + pred_y**2)**0.5
    target_amp = (target_x**2 + target_y**2)**0.5
    error_map = (pred_amp - target_amp)**2
    loss = (error_map * (trimap==128)).sum()
    return loss





if __name__ == "__main__":
    img = np.zeros((10,10)).astype(float)[np.newaxis, np.newaxis, :, :]
    img = torch.FloatTensor(img)
    sigma = 1.4
    theGaussGradient = gaussGradient(sigma)
    a, b = theGaussGradient(img)

    pred = np.zeros((10,10)).astype(float)[np.newaxis, np.newaxis, :, :]
    pred = torch.FloatTensor(pred)
    target = np.ones((10,10)).astype(float)[np.newaxis, np.newaxis, :, :]
    target = torch.FloatTensor(target)
    trimap = (np.ones((10,10)) * 128).astype(float)[np.newaxis, np.newaxis, :, :]
    trimap = torch.FloatTensor(trimap)
    loss = compute_gradient_loss(pred, target, trimap, theGaussGradient)
    pdb.set_trace()

