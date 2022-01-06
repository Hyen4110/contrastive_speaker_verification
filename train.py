import nsml
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as custom_optim
from torch.cuda.amp import GradScaler

from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning import losses, miners, reducers, distances

from loss import angleproto, aamsoftmax, contrastive, arcmargin
from infer import bind_model
from models import patches, resnet, cnnlstm
from utils import * 



####################################################################
#
# set model /criterion/ optimizer
#
####################################################################

def get_model(config):
    if config.version == 1:
        model = cnnlstm.SpeechRecognitionModel(n_cnn_layers = 3,
                                        n_rnn_layers = 5,
                                        rnn_dim = 512,
                                        n_class = 1,  # 1 when BCEloss, 2 when
                                        n_feats = 128,
                                        stride = 2,
                                        dropout = config.dropout_p)
    if config.version == 2:
        model = resnet.resnet34()

    elif config.version == 3:
        model = resnet.resnet34triplet()

    elif config.version == 4:
        model = patches.ConvMixer(128, 34)
    
    elif config.version == 5:
        model = resnet.resnet34Contrastive()

    elif config.version == 7:
        model = resnet.ResNetSE_concat()

    elif config.version == 9:
        model = resnet.ResNetSEContrastive()
    
    elif config.version == 10:
        model = resnet.ResNetSE_arcloss()

    return model
    
def get_crit(config):
    if config.version in [1, 2, 4, 7]:
        criterion = nn.BCEWithLogitsLoss()

    if config.version in [5, 9]:
        criterion = contrastive.contrastive_loss()
    
    if config.version == 8:
        criterion = angleproto()
    
    if config.version == 3:
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low = 0)
        criterion_ = losses.TripletMarginLoss(margin = 0.1,
                                            distance = distance,
                                            reducer = reducer,
                                            triplets_per_anchor = "all")

        mining_func_easy = miners.TripletMarginMiner(margin = 0.2, 
                                                    distance = distance, 
                                                    type_of_triplets = "easy")
        mining_func_semihard = miners.TripletMarginMiner(margin = 0.2, 
                                                        distance = distance, 
                                                        type_of_triplets = "semihard")
        mining_func_hard = miners.TripletMarginMiner(margin = 0.2, 
                                                    distance = distance, 
                                                    type_of_triplets = "hard")
        mining_funcs = {"mining_func_easy": mining_func_easy,
                        "mining_func_semihard": mining_func_semihard,
                        "mining_func_hard": mining_func_hard}
        criterion = {"criterion" : criterion_,
                    "mining_funcs" : mining_funcs}

    if config.version == 10:
        criterion = torch.nn.CrossEntropyLoss()

 
    if config.version == 30:
        mining_funcs = miners.MultiSimilarityMiner(epsilon=0.1)
        criterion_ = losses.ArcFaceLoss(num_classes=847, embedding_size=400, margin=28.6, scale=8)
        criterion = {"criterion" : criterion_,
                    "mining_funcs" : mining_funcs}
    return criterion

def get_optimizer(config, model):
    scaler = GradScaler()
    if config.version == 10:
        margin = arcmargin.ArcMarginProduct(in_feature = 128, 
                                            out_feature = 2,
                                            m = 1.5)

        optimizer = custom_optim.RAdam([{'params':model.parameters()},
                                        {'params':margin.parameters()}], lr = config.learning_rate)

    elif config.version == 30:
        optimizer = {'params' : custom_optim.RAdam(model.parameters(), lr = config.learning_rate),
                    'loss_optimizer': custom_optim.RAdam(model.parameters(), lr = config.learning_rate)}

    else:
        optimizer = custom_optim.RAdam(model.parameters(), lr = config.learning_rate)

    return optimizer, scaler

def initiate(config, train_loader, valid_loader, mfcc_source):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(config).to(device)
    optimizer, scaler = get_optimizer(config, model).to(device)
    criterion = get_crit(config)

    settings = {'model': model,
                'scaler' : scaler,
                'optimizer': optimizer,
                'criterion': criterion}

    # related to nsml system
    bind_model(model = model, parser = config)

    if config.pause: # related to nsml system
        nsml.paused(scope=locals())

    return train_model(settings, config, train_loader, valid_loader)


####################################################################
#
# training and evaluation scripts
#
####################################################################

def train_model(settings, config, train_loader, valid_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = settings['model']
    scaler = settings['scaler']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    def train(epoch, model, optimizer, criterion, scaler):
        avg_cost = 0
        avg_acc = []
        avg_label = []
        total_batch = math.ceil(len(train_loader))

        model.train()

        if config.version == 1 :
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                
                    X = batch['X'].to(device)
                    Y = batch['Y'].to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X)
                        cost = criterion(hypothesis, Y)
                        cost += l_norm(model, l_norm='L1')
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    config.max_grad_norm,)
                    scaler.scale(cost).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    avg_cost += cost / total_batch
                    tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()
                    avg_acc += tmp.tolist()
                    avg_label += Y.detach().cpu().numpy().tolist()

            return avg_cost, avg_acc, avg_label, model

        elif config.version in [2, 4, 7] :
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    Y = batch['Y'].to(device).view(-1)  # |Y| = 64
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2)
                        cost = criterion(hypothesis, Y)


                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                config.max_grad_norm,)
                    scaler.scale(cost).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    avg_cost += cost / total_batch
                    tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()

                    avg_acc += tmp.tolist()
                    avg_label += Y.detach().cpu().numpy().tolist()

            return avg_cost, avg_acc, avg_label, model

        
        elif config.version == 3:
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    X = batch['X'].to(device) # padded_mfcc ([300, 1, 128, 1000])
                    Y = batch['Y'].to(device) # label(speaker_id) ([300])

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X) # embedding값 (300, 400)
                        
                        # tripletloss : 갈수록 어려운 sample(margin 가까이 있는) 학습!
                        if epoch < 100 :
                            mining_func = criterion["mining_funcs"]["mining_func_semihard"]
                        else :
                            mining_func = criterion["mining_funcs"]["mining_func_hard"]

                        # indices_tuple: (ancor_idx, positive_idx, negative_idx) -> len은 463033, 14918 등 그때그때 다름
                        indices_tuple = mining_func(hypothesis, Y)
                        cost = criterion["criterion"](hypothesis, Y, indices_tuple)

                    scaler.scale(cost).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    config.max_grad_norm,)
                    scaler.step(optimizer)
                    scaler.update()

                    avg_cost += cost / total_batch
                
                return model, avg_cost

        elif config.version in [5, 9]:
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    Y = batch['Y'].to(device).view(-1)  # |Y| = 64

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2) # hypothesis == distance
                        cost = criterion(hypothesis, Y)

                    scaler.scale(cost).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    avg_cost += cost / total_batch
                    # tmp = torch.round(torch.sigmoid(
                    #     hypothesis)).detach().cpu().numpy()
                    # avg_acc += tmp.tolist()
                    # avg_label += Y.detach().cpu().numpy().tolist()
                return avg_cost, model

        elif config.version == 10:
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    Y = batch['Y'].to(device).view(-1)  # |Y| = 64

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2) # hypothesis == distance
                        cost = criterion(hypothesis, Y)

                    scaler.scale(cost).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    avg_cost += cost / total_batch
                    # tmp = torch.round(torch.sigmoid(
                    #     hypothesis)).detach().cpu().numpy()
                    # avg_acc += tmp.tolist()
                    # avg_label += Y.detach().cpu().numpy().tolist()
                return avg_cost, model

 
    def valid(epoch, model, criterion, test=False):
        loader = valid_loader
        total_batch = math.ceil(len(loader))
        val_cost = 0
        val_acc = []
        val_label = []

        model.eval()           
        with torch.no_grad():

            if config.version == 1:
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        X = batch['X'].to(device)
                        Y = batch['Y'].to(device)

                        with torch.cuda.amp.autocast():
                            hypothesis = model(X)
                            cost = criterion(hypothesis, Y)

                        val_cost += cost/total_batch
                        tmp = torch.round(torch.sigmoid(
                            hypothesis)).detach().cpu().numpy()
                        val_acc += tmp.tolist()
                        val_label += Y.detach().cpu().numpy().tolist()
                return model, config

            elif config.version in [2, 4, 7]:
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        Y = batch['Y'].to(device)
                        X_1 = batch['X_1'].to(device)
                        X_2 = batch['X_2'].to(device)

                        with torch.cuda.amp.autocast():
                            hypothesis = model(X_1, X_2)
                            cost = criterion(hypothesis, Y)

                        val_cost += cost/total_batch
                        tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()

                        val_acc += tmp.tolist()
                        val_label += Y.detach().cpu().numpy().tolist()

                return model, config
          
            elif config.version == 3:
                acc_by_threshold = {'thr_0.6':0, 'thr_0.7': 0, 'thr_0.8': 0,'thr_0.9': 0}

                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        Y = batch['Y'].to(device)
                        X_1 = batch['left'].to(device)
                        X_2 = batch['right'].to(device)
                        
                        # threshold 에 따라 inference 실행
                        for threshold_key in acc_by_threshold.keys():
                            threshold_num = float(threshold_key.split("_")[1])
                            match_finder = MatchFinder(distance=distances.CosineSimilarity(), threshold = threshold_num)
                            inference_model = InferenceModel(model, match_finder=match_finder)
                            acc_by_threshold[threshold_key] += round(sum(inference_model.is_match(X_1, X_2)==Y.detach().to('cpu').numpy())/(len(Y)*total_batch),3)

                        val_acc = round(sum(acc_by_threshold.values())/len(acc_by_threshold.keys()),3)
                        
                    print(f'acc_by_threshold: {acc_by_threshold}')
                    print(f'val_acc : {val_acc} ')
        
                best_idx = np.argmax(np.array(list(acc_by_threshold.values())))
                best_thresold = float(str(list(acc_by_threshold.keys())[best_idx]).split("_")[1])
                print(f"best threshold: {list(acc_by_threshold.keys())[best_idx]} ({list(acc_by_threshold.values())[best_idx]}) ")
                return model, config, best_thresold

            elif config.version in [5, 9]:
                acc_by_threshold = {'thr_0.6':0, 'thr_0.7': 0, 'thr_0.8': 0,'thr_0.9': 0}

                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        Y = batch['Y'].to(device)
                        X_1 = batch['left'].to(device)
                        X_2 = batch['right'].to(device)
                        
                        # threshold 에 따라 inference 실행
                        for threshold_key in acc_by_threshold.keys():
                            threshold_num = float(threshold_key.split("_")[1])
                            match_finder = MatchFinder(distance=distances.CosineSimilarity(), threshold = threshold_num)
                            inference_model = InferenceModel(model, match_finder=match_finder)
                            acc_by_threshold[threshold_key] += round(sum(inference_model.is_match(X_1, X_2)==Y.detach().to('cpu').numpy())/(len(Y)*total_batch),3)

                        val_acc = round(sum(acc_by_threshold.values())/len(acc_by_threshold.keys()),3)
                        
                    print(f'acc_by_threshold: {acc_by_threshold}')
                    print(f'val_acc : {val_acc} ')
        
                best_idx = np.argmax(np.array(list(acc_by_threshold.values())))
                best_thresold = float(str(list(acc_by_threshold.keys())[best_idx]).split("_")[1])
                print(f"best threshold: {list(acc_by_threshold.keys())[best_idx]} ({list(acc_by_threshold.values())[best_idx]}) ")
                
                return model, config, best_thresold

#################################################################################
#                Let's Start training / validating / testing
#################################################################################
    
    for epoch in range(1, config.epochs+1):
        train(epoch, model, optimizer, criterion, scaler) # trian
        
        # valid
        if config.version in [3, 5, 9, 10]:
            model, config = valid(epoch, model, criterion, test = False) # valid
            dict_for_infer = {'model': model.state_dict(),
                              'config': config,}
        else:
            model, config, best_threshold = valid(epoch, model, criterion, test = False) # valid
            dict_for_infer = {'model': model.state_dict(),
                              'config': config,
                              'best_threshold': best_threshold}
        nsml.save(epoch)

