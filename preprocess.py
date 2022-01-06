import numpy as np
import pandas as pd
import random
import tqdm
from sklearn.preprocessing import LabelEncoder

import soundfile
import librosa

def make_pair_set(df, iteration=8):
    '''
    result : dataframe : left, right, label

        |     |  left  |  right  |  label  |
        |  1  |  135   |  156    |    0    |
        |  2  |  11    |  33     |    1    |
                   ...

    result에 있는 left : 135의 뜻은
    밑에 df의 135번째 row를 뜻합니다.
    하지만 저 df의 mfcc는 source dict에 저장하였습니다.
    source[135]를 하면 left 135의 mfcc가 뽑혀나옵니다.
    -> result가 split 되어서 dataset에 source와 같이 들어갑니다.


    df : filename, speaker, pick

        |      |  filename    | speaker  | pick  |
        |   1  | root/001.wav | idx0001  |   1   |
        |   2  | root/002.wav | idx0001  |   1   | 
                 ...

    pick 뜻은 몇번 뽑혓나 볼려고- 공평하게 장가 시집 보내려고

    df는 60,000개의  row를 갖고 있음.


    '''
    df['pick'] = 0
    # df has [file_name, speaker, pick]

    result = []  # 쌍 {'left':left, "right":right, 'label': 1 or 0}
    np.random.seed(42)

    speaker_order = df.groupby("speaker").count().sort_values("pick").reset_index().speaker

    for _ in tqdm.tqdm(range(iteration)):
        # df = df.sample(frac=1).reset_index(drop=True)
        # speaker_order = df.speaker.unique()
        for sp in speaker_order:  # 0 ~ 10
            in_tmp = df[df.speaker == sp]
            in_in = in_tmp.sample(frac=0.55, replace=False) # 0.66이 0.4: 0.6정도 비율이였음.

            # 0.66 국내로, 0.33 외국으로 가는거지
            # 짝수로 맞추기
            if len(in_in) % 2 != 0:
                in_in = in_in[:-1]

            # 국내로
            in_in_idx = list(in_in.index)
            in_out_idx = [idx for idx in list(
                in_tmp.index) if idx not in in_in_idx]

            np.random.shuffle(in_in_idx)
            np.random.shuffle(in_out_idx)

            df.pick.loc[in_in_idx] = df.pick.loc[in_in_idx] + 1
            df.pick.loc[in_out_idx] = df.pick.loc[in_out_idx] + 1
            for left, right in zip(in_in_idx[::2], in_in_idx[1::2]):
                result.append(
                    {'left_path': left, 'right_path': right, 'label': 1})

            # 외국으로
            out_tmp = df[df.speaker != sp]
            # globaly pick 본거지... 외국에서 가장 인기 없는 친구한테 배정
            out_min = min(out_tmp.pick)
            out_min_idx_1 = list(out_tmp.loc[out_tmp.pick == out_min].index)

            if len(out_min_idx_1) < len(in_out_idx):
                out_min_idx_2 = list(
                    out_tmp.loc[out_tmp.pick == out_min+1].index)
                np.random.shuffle(out_min_idx_2)

                out_chosen_idx = out_min_idx_1 + \
                    out_min_idx_2[:(len(in_out_idx)-len(out_min_idx_1))]

            else:
                np.random.shuffle(out_min_idx_1)
                out_chosen_idx = out_min_idx_1[:len(in_out_idx)]

            for left, right in zip(in_out_idx, out_chosen_idx):
                result.append(
                    {'left_path': left, 'right_path': right, 'label': 0})
                df.pick.iloc[right] = df.pick.iloc[right] + 1

    result = pd.DataFrame(result).drop_duplicates()

    print(f'label 1 : {sum(result.label == 1)/len(result)}')
    print(f'label 0 : {sum(result.label == 0)/len(result)}')

    return result, df

def wav2image_tensor(path, config,sr=16000, n_mfcc=128, n_fft=400, hop_length=100, max_len=1000):
    '''
    여기 바꾸면 custom.wave2image도 바꿔야함.
    
    '''
    if not config.mel: # if not False (mel이 아니면)
        audio, _ = soundfile.read(path)
        audio, _ = librosa.effects.trim(audio)
        
        if config.agument:
            first = np.random.choice(['agument', 'no'], 1)[0]
            if first == 'agument':
                augment = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                ])
                audio = augment(samples=audio, sample_rate=sr)  

        mfcc = librosa.feature.mfcc(
            audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, max_len).reshape(
            1, n_mfcc, max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

def loadWAV(filename, max_frames=600, evalmode=False, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audio, _ = librosa.effects.trim(audio)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats,axis=0).astype(np.float)

    return feat # (1, 32240) // eval mode에서는 (10, 32240) 근데 왜 10배하는지 모르겟음.