
import nsml

import os
import pandas as pd
import numpy as np
import pickle
import warnings
from tqdm import tqdm

import torch
import torch_optimizer as custom_optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning import losses, miners, reducers, distances

from data_loader import get_loaders


def calculate_acc(distance, thr, y=None, mode='train'):
    # distances : list
    # thr : float
    # y : tensor
    thr = float(thr)
    distance = np.array(distance)

    if mode == 'train':
        y = np.array(y)
        # print(f'distance : {distance[:10]}')
        # print(f'threshold : {thr}')
        # print(f'inside calculate_acc y looks like:{y[:10]}')
        result = []
        for i in distance:
            if i < thr:
                result.append(1)
            else:
                result.append(0)
        result = np.array(result)
        # print(f'distance after booling : {distance[:10]}')
        # print(f'distance result : {distance[y==1][:10]}')
        return result # array를 내뱉고 계산은 밖에서 햇네

    else:
        result = []
        for i in distance:
            if i < thr:
                result.append(1)
            else:
                result.append(0)
        result = np.array(result)

        return result

def calculate_ys(lf, rt, threshold):

    mu = np.sum(np.mean(np.concatenate((lf, rt),0)), 0)
    mu = np.expand_dims(mu, 0)

    featureLs = lf - mu
    featureRs = rt - mu
    featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
    featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

    scores = np.sum(np.multiply(featureLs, featureRs), 1)
    for idx,score in enumerate(scores):
        if score > threshold:
            scores[idx] = 1
        else:
            scores[idx] = 0

    return scores


def bind_model(model, config):
    config.mode = 'infer'
    device = torch.device("cuda:0")

    def save_checkpoint(checkpoint, dir):
        torch.save(checkpoint, os.path.join(dir))

    # save trained model 
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True) # directory
        save_dir = os.path.join(dir_name, 'checkpoint')
        save_checkpoint(dict_for_infer, save_dir)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f: 
            pickle.dump(dict_for_infer, f)

        print("trained model saved!")

    # load trained model
    def load(dir_name):
        save_dir = os.path.join(dir_name, 'checkpoint')
        global checkpoint
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model'])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        print("trained model loaded!")

    def infer(config):
        test_loader = get_loaders(config)
        kwargs = {'num_workers': 3, 'pin_memory': True}

        model.eval()
        preds = []
        # print(f'config.version : {config.version}') # print model version

        if config.version == 1:
            for batch in test_loader:
                X = batch['X'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = model(X)
                    pred = torch.tensor(torch.round(torch.sigmoid(pred)), dtype=torch.long).cpu().numpy()
                    preds += list(pred)

        elif config.version in [2, 4, 7]:
            for batch in test_loader:
                X_1 = batch['X_1'].to(device)
                X_2 = batch['X_2'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = model(X_1, X_2)


        elif config.version == 3:
            best_threshold = dict_for_infer['best_threshold']
            match_finder = MatchFinder(distance = distances.CosineSimilarity(), 
                                       threshold = best_threshold)
            inference_model = InferenceModel(model, match_finder = match_finder)

            for batch in test_loader:
                X_1 = batch['X_1'].to(device)
                X_2 = batch['X_2'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = inference_model.is_match(X_1, X_2).astype(np.int64)  # True/False -> int로
                    preds += list(pred)

        elif config.version in [5, 9] :
            distances = []
            best_th = dict_for_infer['best_threshold']
            
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)

                    with torch.cuda.amp.autocast():
                        x1 = torch.from_numpy(X_1).float().to(device)
                        x2 = torch.from_numpy(X_2).float().to(device)
                        with torch.no_grad():
                            dist = model(x1, x2).cpu().numpy().flatten()
                            
                    distances += dist.tolist()
                distances = np.array(distances)
                preds = calculate_acc(distances, thr = best_th, mode = 'infer').tolist()

        elif config.version == 10:
            best_th = dict_for_infer['best_threshold']
            with torch.no_grad():
                cnt = 0
                for batch in tqdm(test_loader):
                    Y = batch['Y'].to(device)
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    # with torch.cuda.amp.autocast():
                    res = model(X_1, X_2)
                    res = res[:,128:]
                    lf = res[:,:128].detach().cpu().numpy()
                    rt = res[:,128:].detach().cpu().numpy()
                    Y = Y.detach().cpu().numpy()

                    if cnt == 0:
                        lfs = lf
                        rts = rt
                        ys = Y
                    else:
                        lfs = np.concatenate((lfs, lf), 0)
                        rts = np.concatenate((rts, rt), 0)
                        ys = np.concatenate((ys, Y), 0)

                preds = calculate_ys(lfs, rts, best_th).tolist()

        # DO NOT CHANGE: They are reserved for nsml
        # 리턴 결과는 [(prob, pred)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # 위의 포맷에서 prob은 리더보드 결과에 확률의 값은 영향을 미치지 않습니다(pred만 가져와서 채점).
        # pred에는 예측한 binary 혹은 1에 대한 확률값을 넣어주시면 됩니다.
        prob = [1]*len(preds)

        return list(zip(prob, preds))

        nsml.bind(save=save, load=load, infer=infer)