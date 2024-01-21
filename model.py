# -*- coding: utf-8 -*-

import functools
import math
# from typing_extensions import final

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import init

EPS = 1e-12
bias = True

class ResBlock_dense(nn.Module):
    def __init__(self, fin=128, fout=128):
        super(ResBlock_dense, self).__init__()
        self.conv0 = nn.Conv2d(fin, fout, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(fout, fout, 3, 1, 1, bias=bias)
    def forward(self, x):
        # x[0]: concat; x[1]: fea
        # fea = x[0] * (cond + 1)
        fea = F.leaky_relu(self.conv0(x[0]), 0.2, inplace=True)
        # fea = fea * (cond + 1)
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        fea = fea * (x[2] + 1)
        # x0 = x[0] + fea
        x0 = x[1] + fea
        return (x0, x[1], x[2])

class ResBlock_conv_dense(nn.Module):
    def __init__(self, fin=128, fout=128):
        super(ResBlock_conv_dense, self).__init__()
        self.conv0 = nn.Conv2d(fin, fout, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(fout, fout, 3, 1, 1, bias=bias)
    def forward(self, x):
        fea = F.leaky_relu(self.conv0(x[0]), 0.2, inplace=True)
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        x0 = x[1] + fea
        return (x0, x[1], x[2])


class CreateGenerator(nn.Module):
    def __init__(self, inchannels=8, grads=3):
        super(CreateGenerator, self).__init__()
        f = 32
        # ms and pan images
        self.encoder21 = nn.Sequential(nn.Conv2d(inchannels, f, 3, 1, 1, bias=bias),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 1, 1, bias=bias))
        self.encoder22 = nn.Sequential(nn.Conv2d(1, f, 3, 1, 1, bias=bias),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 1, 1, bias=bias))

        self.encoder41 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))
        self.encoder42 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))
        self.encoder31 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))
        self.encoder32 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(f, f, 3, 2, 1, bias=bias))

        self.conv0 = nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias)

        # fusion stage 
        self.CondNet = nn.Sequential(nn.Conv2d(grads, f, 3, 1, 1, bias=bias),
                                     nn.LeakyReLU(0.2, True),
                                     nn.Conv2d(f, f, 3, 1, 1, bias=bias),
                                     nn.LeakyReLU(0.2, True))
        res_branch = []
        self.cond_stage1 = nn.Sequential(nn.Conv2d(f, f, 4, 2, 1, bias=bias),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(f, f*2, 4, 2, 1, bias=bias))

        k = 0
        for i in range(4):
            # k = k + f*2
            res_branch.append(ResBlock_dense(f*2, f * 2))
            # res_branch.append(ResBlock_conv_dense(f*2, f * 2))

        self.res_branch = nn.Sequential(*res_branch)

        # ---------------------------------upsampling stage---------------------------------
        # ----------------------------------------------------------------------------------
        self.cond_stage2 = nn.Sequential(nn.Conv2d(f, f, 4, 2, 1, bias=bias),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(f, f*2, 3, 1, 1, bias=bias))



        self.HR_branch = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias))
        # self.HR_branch = nn.Sequential(nn.Conv2d(f * 2, f * 2 * 4, 3, 1, 1, bias=bias),
        #                                 nn.PixelShuffle(2),
        #                                 nn.LeakyReLU(0.2))

        res_branch1 = []
        for i in range(1):
            res_branch1.append(ResBlock_dense(f*2, f * 2))
        self.res_branch1 = nn.Sequential(*res_branch1)

        # ---------------------------------upsampling stage---------------------------------
        # ----------------------------------------------------------------------------------
        self.cond_stage3 = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1, bias=bias),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(f, f*2, 3, 1, 1, bias=bias))


        self.HR_branch2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                        nn.Conv2d(f * 2, f * 2, 3, 1, 1, bias=bias))
        # self.HR_branch2 = nn.Sequential(nn.Conv2d(f * 2, f * 2 * 4, 3, 1, 1, bias=bias),
        #                                 nn.PixelShuffle(2),
        #                                 nn.LeakyReLU(0.2))

        self.conv1 = nn.Conv2d(f * 4, f * 2, 3, 1, 1, bias=bias)

        res_branch2 = []
        for i in range(1):
            res_branch2.append(ResBlock_dense(f * 2, f * 2))
            # res_branch2.append(ResBlock_conv_dense(f * 2, f * 2))
        self.res_branch2 = nn.Sequential(*res_branch2)

        self.rec = nn.Conv2d(f * 2, inchannels, 3, 1, 1, bias=bias)
    def forward(self, ms_in, pan_in, grads, up_ms):
        # conditions
        cond = self.CondNet(grads)
        cond1 = self.cond_stage1(cond)
        cond2 = self.cond_stage2(cond)
        cond3 = self.cond_stage3(cond)
        # encoder
        ms_encoder = self.encoder21(up_ms)
        pan_encoder = self.encoder22(pan_in)
        ms_up1 = self.encoder31(ms_encoder)
        pan_up1 = self.encoder32(pan_encoder)
        ms_up2 = self.encoder41(ms_up1)
        pan_up2 = self.encoder42(pan_up1)
        concat = torch.cat((ms_up2, pan_up2), dim=1)
        fea = self.conv0(concat)
        feas = [fea]
        # fusion

        l1 = 1
        l2 = 1
        l3 = 1

        for i in range(l1):
            fea = self.res_branch((fea, feas[0], cond1))[0]
            # fea = self.res_branch((fea, feas[0], fea))[0]

        # upsampling
        fea = self.HR_branch(fea)

        feas.append(fea)
        for i in range(l2):
            fea = self.res_branch1((fea, feas[1], cond2))[0]
            # fea = self.res_branch1((fea, feas[1], fea))[0]
        fea = self.HR_branch2(fea)
        fea = torch.cat((ms_encoder, pan_encoder, fea), dim=1)
        fea = self.conv1(fea)

        feas.append(fea)
        for i in range(l3):
            fea = self.res_branch2((fea, feas[2], cond3))[0]
            # fea = self.res_branch2((fea, feas[2], fea))[0]
        rec = self.rec(fea)+up_ms
        return rec

