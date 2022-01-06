#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
import sklearn
import librosa


def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def max_len_check(df):
    # To check length of wav  for EDA
    def len_check(path, sr=16000, n_mfcc=100, n_fft=400, hop_length=160):
        audio, sr = librosa.load(path, sr=sr)
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        return mfcc.shape[1]

    left_len = df['left_path'].apply(lambda x: len_check(x))
    right_len = df['right_path'].apply(lambda x: len_check(x))

    left_max_len = left_len.max()
    right_max_len = right_len.max()

    return (max(left_max_len, right_max_len))

def l_norm(model, l_norm='L1'):
    # set l_norm
    if l_norm == 'L1':
        L = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                L = L + torch.norm(param, 1)
        L = 10e-4 * L
    elif l_norm == 'L2':
        L = sum(p.pow(2.0).sum() for p in model.parameters())
    return L