from nsml import DATASET_PATH

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.model_selection import train_test_split

import soundfile
import librosa
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from preprocess import *


#############################################################################
#                       Dataset Class
#############################################################################
class CustomDataset(Dataset):
    def __init__(self, left_path, right_path, max_len=1000, config=None, label=None, mode='train'):
        self.left_path = left_path
        self.right_path = right_path
        self.max_len = max_len
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

    def __len__(self):
        return len(self.left_path)

    def wav2image_tensor(self, path):
        audio, sr = librosa.load(path, sr=self.sr)
        audio, _ = librosa.effects.trim(audio)
        mfcc = librosa.feature.mfcc(
            audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, self.max_len).reshape(
            1, self.n_mfcc, self.max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.wav2image_tensor(left_path)
        right_padded_mfcc = self.wav2image_tensor(right_path)

        padded_mfcc = torch.cat([left_padded_mfcc, right_padded_mfcc], dim=0)

        if self.mode == 'train':
            label = self.label[i]

            if self.config.BCL:
                label = torch.tensor(label, dtype=torch.float)
            else:
                label = torch.tensor(label, dtype=torch.long)

            return {
                'X': padded_mfcc,
                'Y': label
            }
        else:
            return {
                'X': padded_mfcc
            }


class CustomDataset_2output(Dataset):
    def __init__(self, left_path, right_path, config=None, label=None, mode='train', max_len=1000):
        self.left_path = left_path
        self.right_path = right_path
        self.max_len = max_len
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

    def __len__(self):
        return len(self.left_path)

    def wav2image_tensor(self, path):
        if self.config.version == 1 or self.config.version == 2 or self.config.version == 3 or self.config.version == 4 or self.config.version == 5 or self.config.version == 6 or self.config.version == 8 or self.config.version == 30:
            audio, sr = librosa.load(path, sr=self.sr)
            audio, _ = librosa.effects.trim(audio)
            mfcc = librosa.feature.mfcc(
                audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
                (a, np.zeros((a.shape[0], i-a.shape[1]))))
            padded_mfcc = pad2d(mfcc, self.max_len).reshape(
                1, self.n_mfcc, self.max_len)  # 채널 추가
            padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

            return padded_mfcc  # [bs, 1, 128, 1000] or [bs, 1, 1000, 128]

        if self.config.version == 7 or self.config.version == 9 or self.config.version == 10:
            audio = loadWAV(path)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=40)
                    audio = torch.FloatTensor(audio)
                    x = torchfb(audio) + 1e-6
                    x = x.log()
                    x = x.squeeze(0).detach() #[1,40,202] -> [40,202]
                    # x = x.detach()
            return x  #[40, 202]

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.wav2image_tensor(left_path)
        right_padded_mfcc = self.wav2image_tensor(right_path)

        if self.mode == 'train':
            label = self.label[i]

            if self.config.BCL:
                label = torch.tensor(label, dtype=torch.float)
            else:
                label = torch.tensor(label, dtype=torch.long)

            return {
                'X_1': left_padded_mfcc,
                'X_2': right_padded_mfcc,
                'Y': label
            }
        else:
            return {
                'X_1': left_padded_mfcc,
                'X_2': right_padded_mfcc,
            }

class FinalDataset2output(Dataset):
    def __init__(self, left_path, right_path, source, config=None, label=None, mode='train'):
        self.left_path = left_path
        self.right_path = right_path
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label
            self.dropout1 = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )
        if self.mode == 'valid':
            self.label = label


    def __len__(self):
        return len(self.left_path)

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.source[left_path]
        right_padded_mfcc = self.source[right_path]

        if self.mode == 'train':
            left_padded_mfcc = self.dropout1(left_padded_mfcc)
            right_padded_mfcc = self.dropout1(right_padded_mfcc)

        label = self.label[i]

        if self.config.BCL:
            # BCL일때는 tensor type float을 요구함.
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return {'X_1': left_padded_mfcc,
                'X_2': right_padded_mfcc,
                'Y': label
                 }

class FinalDatasetTriple(Dataset):
    def __init__(self, df, source, config=None, mode='train'):
        ''' df
        |    |  filename  |  speaker |  pick  |
        |  1 |  ...001.wav|    idx01 |        |
        |  2 |  ...002.wav|    idx02 |        |
                        ....
        return -> {X, Y}이다.
        '''
        self.df = self.sampling(df).reset_index(drop = True)
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.dropout1 = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )

        print(f'length of final dataset : {len(self.df)}')

    def sampling(self, df, max_num = 200, min_num=100):
        picked_idx = []
        for speaker in df.speaker.unique():
            tmp = df[df.speaker == speaker]
            if len(tmp) > max_num:
                picked_idx += np.random.choice(tmp.index.tolist(), max_num, replace = False).tolist()
            elif 5*len(tmp) < min_num:
                numAdd = 100-len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), numAdd, replace = True).tolist()
            else: # 5배보다 작고 
                numAdd = max_num - len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), numAdd, replace = True).tolist()
        
        return df.loc[picked_idx]

    def stretch_sound(self, data, sr=16000, rate=0.8):# stretch 해주는 것 테이프 늘어진 것처럼 들린다.
        stretch_data = librosa.effects.time_stretch(data, rate)
        return stretch_data

    def reverse_sound(self, data, sr=16000):# 거꾸로 재생
        data_len = len(data)
        data = np.array([data[len(data)-1-i] for i in range(len(data))])
        return data

    def wave2image_tensor(self,path, pre,sr=16000, n_mfcc=128, n_fft=400, hop_length=100, max_len=1000):
        audio, _ = soundfile.read(path)
        audio, _ = librosa.effects.trim(audio)

        # if preprocess == 'No':
        #     pass
        # elif preprocess == 'reverse':
        #     audio = self.reverse_sound(audio)
        # else:
        #     audio = self.stretch_sound(audio, rate = preprocess)

        if pre == 'agument':
            augment = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                              ])
            audio = augment(samples=audio, sample_rate=sr)


        mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, max_len).reshape(
            1, n_mfcc, max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        first = np.random.choice(['agument', 'no'], 1)[0]
        # if first == 'No':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'No')
        # elif first == 'stretch':
        #     preprocess = np.random.uniform(0.7, 1.3, 1)
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess=preprocess)
        # elif first == 'reverse':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'reverse')
        # elif first == 'zero':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'No')
        #     padded_mfcc = self.dropout1(padded_mfcc)
        padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], pre=first)
        
        label = self.df['speaker'].iloc[i]
        label = torch.tensor(label, dtype=torch.long)

        return {'X': padded_mfcc,
                'Y': label}

class FinalDatasetTriple_infer(Dataset):
    def __init__(self, left_path, right_path, source, df, config=None, label=None, mode='train'):
        
        self.left_path = left_path
        self.right_path = right_path
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

            left_label = left_path.values
            right_label = right_path.values
            self.left_speaker_label = df.loc[left_label]['speaker'] # index = left_label
            self.right_speaker_label = df.loc[right_label]['speaker'] # index = right_label


    def __len__(self):
        return len(self.left_path)

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.source[left_path]
        right_padded_mfcc = self.source[right_path]


        if self.mode == 'train':
            label = self.label.iloc[i]
            l_label = self.left_speaker_label.iloc[i]
            r_label = self.right_speaker_label.iloc[i] 


        if self.config.BCL:
            # BCL일때는 tensor type float을 요구함.
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)


        if self.mode == 'train':
            return {
                'left': left_padded_mfcc,
                'right': right_padded_mfcc,
                'l_label' : l_label,
                'r_label': r_label,
                'Y': label
            }

        else: # inference
            return {
                'left' : left_padded_mfcc,
                'right' : right_padded_mfcc,
            }


#############################################################################
#                       util functions
#############################################################################

# get same count of class
def sampling(df, max_num = 150, min_num = 80):
        picked_idx = []
        for speaker in df.speaker.unique():
            tmp = df[df.speaker == speaker]
            if len(tmp) > max_num:
                picked_idx += np.random.choice(tmp.index.tolist(), max_num, replace = False).tolist()

            elif len(tmp) < min_num:
                # numAdd = min_num - len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), min_num, replace = True).tolist()

            else:
                # numAdd = max_num - len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), max_num, replace = True).tolist()
        return df.loc[picked_idx]


def get_data(config):
    '''
        train_df : root/train/train_data/wav/idx00001, idx00002
                | ind |                   file_name                         |   speaker   |
                ---------------------------------------------------------------------------
                |  0  |  /data/final_speaker/train/train_data/wav/idx00...  |  S000013_1  |
                |  1  |  /data/final_speaker/train/train_data/wav/idx00...  |  S000013_2  |
    '''
    if config.mode == 'train':
        train_path = os.path.join(DATASET_PATH, 'train/train_data')
        root_path = os.path.join(train_path, 'wav')
        train_df = pd.read_csv(os.path.join(train_path, 'train_info'))
        train_df['file_name'] = root_path + '/' + train_df['file_name']

    #infer
    else:
        test_data = pd.read_csv(os.path.join(config.test_path, 'test_data', 'test_data'))
        root_path = os.path.join(config.test_path, 'test_data', 'wav')
        test_df = pd.DataFrame({'left_path': root_path + '/' + test_data['file_name'],
                                 'right_path': root_path + '/' + test_data['file_name_']})

    return train_df if config.mode == 'train' else test_df

def get_mfcc(pickcnt_df, config):
    mfcc_source = dict()  # {1:mfcc, 2:mfcc, ...}
    for idx, row in pickcnt_df.iterrows():
        try:
            mfcc_source[idx]
        except:
            mfcc_source[idx] = wav2image_tensor(row.file_name, config)
    return mfcc_source

def data_split(dataframe, ratio=False):
    '''
    spliting dataset into train(0.9), test(0.1)
    <input>: dataframe
    <return> : [train_array, valid_array]
    '''
    # 200000 samples -> (mfcc)80GB // (mel)84GB...
    if ratio:
        x = dataframe['right_path']
        y = dataframe['label']
        train, test, _, _ = train_test_split(
            x, y, test_size = ratio, random_state = 28, stratify = y)
        dataframe = dataframe.loc[train.index]
    x = dataframe['right_path']
    y = dataframe['label']
    train, test, _, _ = train_test_split(
        x, y, test_size = 0.15, random_state = 28, stratify = y)

    return list(train.index), list(test.index)

#############################################################################
#                       (main) get_loaders
#############################################################################

def get_loaders(config):
    if config.mode == 'train':
        train_df = get_data(config)
        pair_df, filenm_df = make_pair_set(train_df, iteration = config.iteration) 
        # pair_df   : (dataframe) left_id, right_id, label(0/1)
        # filenm_df : (dataframe) filename, speaker, used_count

        # POC -> random sampling by n_sample
        if config.POC:
            pair_df = pair_df.iloc[:config.n_sample]
            # concat for converting pair-file to mfcc at the same time
            tmp_list = pair_df['left_path'].values.tolist() + pair_df['right_path'].values.tolist() # get ids
            filenm_df = filenm_df.loc[tmp_list]
        
        # reference mfcc dictionary
        mfcc_source = get_mfcc(filenm_df, config) 

        # train valid split
        train_index, valid_index = data_split(pair_df)
        valid_df = pair_df.loc[valid_index].reset_index(drop = True)
        train_df = pair_df.loc[train_index].reset_index(drop = True)

        if config.version != 3:
            # {'X_1': left_padded_mfcc, 'X_2': right_padded_mfcc, 'Y': label}
            train_df = FinalDataset2output(left_path = train_df['left_path'],
                                            right_path = train_df['right_path'],
                                            label = train_df['label'],
                                            source = mfcc_source,
                                            mode = 'train',
                                            config = config)

            valid_df = FinalDataset2output(left_path = valid_df['left_path'],
                                            right_path = valid_df['right_path'],
                                            label = valid_df['label'],
                                            source = mfcc_source,
                                            mode = 'valid',
                                            config = config)
        # contrastive learning
        elif config.version == 3:                
            # {'X': padded_mfcc, 'Y': label}
            train_dataset = FinalDatasetTriple(df = filenm_df,
                                        source = mfcc_source,
                                        mode = 'train',
                                        config = config)

            # 'trian'  -> {'left': left_padded_mfcc, 'right': right_padded_mfcc,
            #            'l_label' : l_label, 'r_label': r_label, 'Y': label
            # 'infer' -> {'left' : left_padded_mfcc, 'right' : right_padded_mfcc,}
            valid_dataset = FinalDatasetTriple_infer(left_path = pair_df['left_path'],
                                                    right_path = pair_df['right_path'],
                                                    label = pair_df['label'],
                                                    source  =mfcc_source,
                                                    mode = 'train' if config.POC else 'infer',
                                                    df = filenm_df,
                                                    config = config)

        train_loader = DataLoader(dataset = train_dataset,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    **kwargs)

        valid_loader = DataLoader(dataset = valid_dataset,
                                    batch_size = config.batch_size,
                                    shuffle = True,
                                    **kwargs)


        return train_loader, valid_loader, mfcc_source

    if config.mode == 'infer':
        test_df = get_data(config)

        # {'X_1': left_padded_mfcc, 'X_2': right_padded_mfcc}
        test_dataset = CustomDataset_2output(test_df['left_path'], 
                                    test_df['right_path'], 
                                    max_len = 1000, 
                                    label = None, 
                                    mode = 'test', 
                                    config = config)

        test_loader = DataLoader(dataset = test_dataset,
                                    batch_size = config.batch_size,
                                    shuffle = False,
                                    **kwargs)

        return test_loader