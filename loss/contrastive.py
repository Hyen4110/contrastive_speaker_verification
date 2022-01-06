#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(y, t):
    nonmatch = F.relu(3 - y)  # max(margin - y, 0)
    return torch.mean(t * y**2 + (1 - t) * nonmatch**2)