import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import math

irange = range
# reference : https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def calc_gradient_penalty(netD, real_data, fake_data, use_gpu = True, dec_output=1):
    alpha = torch.rand(real_data.shape[0], 1)
    if len(real_data.shape) == 4:
        alpha = alpha.unsqueeze(2).unsqueeze(3)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if dec_output==2:
        disc_interpolates,_ = netD(interpolates)
    elif dec_output == 3:
        disc_interpolates,_,_ = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_gpu else torch.ones(
                                  disc_interpolates.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def ae_loss(mu, logvar):
    # loss = rec_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return  KLD
