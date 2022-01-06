import nsml
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch_optimizer as custom_optim


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from utils import * 

from pytorch_metric_learning import losses, miners, reducers, distances
from loss import angleproto, aamsoftmax, contrastive, arcmargin
from infer import bind_model
from models import patches, resnet, cnnlstm



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
    scaler = torch.cuda.amp.GradScaler()
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

def initiate(config, train_loader, valid_loader, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer, scaler = get_optimizer(config).to(device)
    model = get_model(config).to(device)
    criterion = get_crit(config).to(device)

    settings = {'model': model,
                'scaler' : scaler,
                'optimizer': optimizer,
                'criterion': criterion}

    # related to nsml system
    bind_model(model = model, parser = config)

    if config.pause: # related to nsml system
        nsml.paused(scope=locals())

    return train_model(settings, config, train_loader, valid_loader, test_loader, scaler)


####################################################################
#
# training and evaluation scripts
#
####################################################################

def train_model(settings, config, train_loader, valid_loader, test_loader, scaler):
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
        loader = test_loader if test else valid_loader

        valid_loss = 0.0
        num_batches = config.n_test // config.batch_size if test else config.n_valid // config.batch_size
        total_embeddings = np.empty((0,config.embedding_size), int)
        total_pred, total_label = [], []

        model.eval()           

        if config.version == 2:
            with torch.no_grad():
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        text = batch['text']
                        token_ids = text[0].squeeze(1).long().to(device) # [128, 57]
                        attention_mask = text[1].squeeze(1).long().to(device) # [128, 57]
                        token_type_ids = text[2].squeeze(1).long().to(device) # [128, 57]

                        label = batch['label'].long().to(device).view(-1) # [128]
                        
                        with torch.cuda.amp.autocast():
                            embeddings, preds = model(token_ids, attention_mask, token_type_ids)
                            loss = crits['crit_ce'](preds, label)

                        valid_loss += loss
                        
                        # append              
                        total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                        total_label += label.detach().cpu().numpy().tolist()
                        
                if test and (epoch >3):
                    print(f" testing Confusion Matrix")
                    print(confusion_matrix(total_label, total_pred))
                f1_score_ = f1_score(total_label, total_pred, average = 'macro')  
                accuracy = accuracy_score(total_label, total_pred)
                print(f"[Epoch {epoch} {'testing' if test else 'validating'}] : loss = {float(valid_loss)/num_batches:.4f}, accuracy = {accuracy:.4f}, f1_score = {f1_score_:.4f}")

            return total_embeddings, total_label
            
        elif config.version in [3,4]:
            
            with torch.no_grad():
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        
                        text = batch['text']
                        token_ids = text[0].squeeze(1).long().to(device) # [128, 57]
                        attention_mask = text[1].squeeze(1).long().to(device) # [128, 57]
                        token_type_ids = text[2].squeeze(1).long().to(device) # [128, 57]

                        label = batch['label'].long().to(device).view(-1) # [128]
                        
                        with torch.cuda.amp.autocast():
                            embeddings, preds = model(token_ids, attention_mask, token_type_ids)
                            loss = crits['crit_ce'](preds, label)

                        valid_loss += loss
                        
                        # append
                        total_embeddings = np.vstack([total_embeddings, embeddings.detach().cpu().numpy()])                        
                        total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                        total_label += label.detach().cpu().numpy().tolist()
                                                    
                print(f"[Epoch {epoch} {'testing' if test else 'validating'}] : loss = {float(valid_loss)/num_batches:.4f}")

            return total_embeddings, total_label
                

#################################################################################
#                Let's Start training / validating / testing
#################################################################################

    if config.version ==2 :
        for epoch in range(1, config.num_epochs+1):
            train(epoch, model, optimizer, criterion, scaler) # trian
            valid(epoch, model, criterion, test=False) # valid
            valid(epoch, model, criterion, test=True) # test

            save_model(config, epoch, model)
                
    else:
        for epoch in range(1, config.num_epochs+1):
            # train with tsne
            embeddings, label = train(epoch, model, optimizer, criterion, scaler) # trian
            label_center = plot_t_SNE(config, epoch, embeddings, label, state='train')  # dictionary {label: center(128,)}           
            
            # valid 
            valid(epoch, model, criterion,test=False)
            
            # test with tsne
            embeddings, label = valid(epoch, model, criterion, test=True)
            get_accacy_similarity(config, epoch, embeddings, label, label_center)
            plot_t_SNE(config, epoch, embeddings, label, state='test')
            
            #save model
            save_model(config, epoch, model) # Model Save
    