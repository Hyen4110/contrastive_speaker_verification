import nsml

import argparse
import pandas as pd
import numpy as np
import pprint
import random

import torch
import train
from data_loader import get_loaders

def define_argparser():
    parser = argparse.ArgumentParser(description='2021 speaker verification contest')    
    parser.add_argument('--version', type = int, default = 3,
                        help = 'version 1 : unimodal with BCLoss and input two speakers\
                                version 2 : siamness ResNet model with BCLoss or siamness transformer model\
                                             with BCLoss and input two speakers\
                                version 3 : ResNet with tripletLoss input one speaker\
                                version 4 : ConvMixer with BCLoss and input two speaker patches are all you need(ICLR 2021) \
                                version 5 : ResNet with contrasive loss and input two speaker \
                                version 7 : thin-SEResNet with anglerLoss and input two speaker \
                                version 9 : thin-SEResNet with contrasive loss input two speaker \
                                version 10 : this SEResNet pretrain with ArcMargin,\
                                            get threshold with CrossEntropyLoss, and input one speaker ')

    # for POC mode                                         
    parser.add_argument('--POC', type = bool, default = False,  
                        help = 'for testing since preprocess is time consuming work')
    parser.add_argument('--n_sample', type = int, default = 500,  
                        help = '(POC) number of test sample ')
    parser.add_argument('--iteration', type = int, default = 2,  
                        help = 'number of iteration in sampling POC mode ')
                        
    # trarining details
    parser.add_argument('--epochs', type = int, default=100, 
                        help = 'number of epochs to train')
    parser.add_argument('--learning_rate',type = float,  default = 1e-4)
    parser.add_argument('--v',type = float,  default = 0.1)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--pause', type  = int, default = 0,
                        help = 'relate to nsml system. barely change')
    parser.add_argument('--max_grad_norm',type = float, default = 5.,
                        help = 'maximum gradient for gradient clipping')
    
    # affects to custom.py/ label should be float -> 1
    parser.add_argument('--BCL', type = bool, default = False,
                        help = 'this is used when target is set to be binary.\
                                 this should be True when version 1,2,4,7')
    parser.add_argument('--iteration', type = int, default = 2,
                        help = 'it works on version 5, 7, 9, 10')
    
    #audio preprocess
    parser.add_argument('--mel', type = bool, default = False,
                        help = 'it works on version 7, 9, 10')
    parser.add_argument('--n_mels', type = int, default = 80,
                        help = 'it works on version 7, 9, 10')  
    
    # etc (not hyper-parameter)
    parser.add_argument('--mode', type = str, default = 'train', \
                    help = 'relate to nsml system.')
    parser.add_argument('--test_path', type = str, default = 'yet', \
                        help = 'path for infer data (in nsml system)')

    config = parser.parse_args()

    return config

def print_config(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))

def set_random_seed(config):
    # set random seed
    torch.manual_seed(config.seed) # for CPU
    torch.cuda.manual_seed(config.seed) # for CUDA
    random.seed(config.seed) 
    torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == '__main__':
    config = define_argparser() 
    set_random_seed(config)
    print_config(config)

    train_loader, valid_loader, mfcc_source = get_loaders(config)
    train.initiate(config, train_loader, valid_loader, mfcc_source)

