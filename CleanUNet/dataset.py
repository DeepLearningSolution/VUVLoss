# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio


class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, unvoiced, voiced, file_id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0, load_vuv=False):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset
        self.root = root
        self.load_vuv = load_vuv  # 是否加载VUV参考数据

        N_clean = len(os.listdir(os.path.join(root, 'training_set/clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'training_set/noisy')))
        assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'training_set/clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/noisy', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/unvoiced', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/voiced', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
        
        elif subset == "testing":
            sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
            _p = os.path.join(root, 'datasets/test_set/synthetic/no_reverb')  # path for DNS
            #_p = os.path.join(root, 'datasets/test_set/real_recordings')  # path for DNS
            
            clean_files = os.listdir(os.path.join(_p, 'clean'))
            noisy_files = os.listdir(os.path.join(_p, 'noisy'))
            
            clean_files.sort(key=sortkey)
            noisy_files.sort(key=sortkey)

            self.files = []
            for _c, _n in zip(clean_files, noisy_files):
                assert sortkey(_c) == sortkey(_n)
                self.files.append((os.path.join(_p, 'clean', _c), 
                                   os.path.join(_p, 'noisy', _n),
                                   None, None))
            self.crop_length_sec = 0
            self.load_vuv = False

        else:
            raise NotImplementedError

    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0], backend="soundfile")
        noisy_audio, sample_rate = torchaudio.load(fileid[1], backend="soundfile")
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        if self.load_vuv and fileid[2] is not None and fileid[3] is not None:
            unvoiced_audio, _ = torchaudio.load(fileid[2], backend="soundfile")
            voiced_audio, _ = torchaudio.load(fileid[3], backend="soundfile")
            unvoiced_audio = unvoiced_audio.squeeze(0)
            voiced_audio = voiced_audio.squeeze(0)
        else:
            unvoiced_audio = None
            voiced_audio = None

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)

        # random crop
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]
            if unvoiced_audio is not None:
                unvoiced_audio = unvoiced_audio[start:(start + crop_length)]
                voiced_audio = voiced_audio[start:(start + crop_length)]
        
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        if unvoiced_audio is not None:
            unvoiced_audio = unvoiced_audio.unsqueeze(0)
            voiced_audio = voiced_audio.unsqueeze(0)
        
        return (clean_audio, noisy_audio, unvoiced_audio, voiced_audio, fileid)

    def __len__(self):
        return len(self.files)


def collate_fn_with_none(batch):
    """
    Custom collate function to handle None values in VUV data
    Returns: (clean, noisy, unvoiced, voiced, fileid_list)
    where unvoiced and voiced can be None if load_vuv=False
    """
    clean_list, noisy_list, unvoiced_list, voiced_list, fileid_list = [], [], [], [], []
    
    for clean, noisy, unvoiced, voiced, fileid in batch:
        clean_list.append(clean)
        noisy_list.append(noisy)
        unvoiced_list.append(unvoiced)
        voiced_list.append(voiced)
        fileid_list.append(fileid)
    
    # Stack clean and noisy (always present)
    clean_batch = torch.stack(clean_list, dim=0)
    noisy_batch = torch.stack(noisy_list, dim=0)
    
    # Handle VUV data (may be None)
    if unvoiced_list[0] is None:
        unvoiced_batch = None
        voiced_batch = None
    else:
        unvoiced_batch = torch.stack(unvoiced_list, dim=0)
        voiced_batch = torch.stack(voiced_list, dim=0)
    
    return clean_batch, noisy_batch, unvoiced_batch, voiced_batch, fileid_list


def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1, load_vuv=False):
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec, load_vuv=load_vuv)                                                       
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False, "collate_fn": collate_fn_with_none}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
    return dataloader


if __name__ == '__main__':
    import json
    with open('./configs/DNS-large-full.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=2, num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=2, num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader: 
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break
    