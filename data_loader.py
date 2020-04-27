import glob

import os



import librosa

import numpy as np

import torch

from sklearn.preprocessing import LabelBinarizer

from torch.utils.data.dataloader import DataLoader

from torch.utils.data.dataset import Dataset



#from preprocess import (FEATURE_DIM, FFTSIZE, FRAMES, SAMPLE_RATE,

 #                       world_features)

from utility import *

import random



class AudioDataset(Dataset):

    """docstring for AudioDataset."""

    def __init__(self, datadir:str):

        super(AudioDataset, self).__init__()

        self.datadir = datadir

        self.files = librosa.util.find_files(datadir, ext='npy')

        self.encoder = LabelBinarizer().fit(styles)

        



    def __getitem__(self, idx):

        p = self.files[idx]

        filename = os.path.basename(p)

        style = filename.split(sep='_', maxsplit=1)[0]

        label = self.encoder.transform([style])[0]

        mid = np.load(p)*1.


        mid = torch.FloatTensor(mid)

        #mid = torch.unsqueeze(mid, 0)

        return mid, torch.tensor(styles.index(style), dtype=torch.long), torch.FloatTensor(label)



    def speaker_encoder(self):

        return self.encoder



    def __len__(self):

        return len(self.files)



def data_loader(datadir: str, batch_size=4, shuffle=True, mode='train', num_workers=0):

    '''if mode is train datadir should contains training set which are all npy files

        or, mode is test and datadir should contains only wav files.

    '''

    dataset = AudioDataset(datadir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    

    return loader

print(list(reversed(styles)))

#ad = AudioDataset('./data/rock_bossanova_funk_RnB')
#print(len(ad))

#data, s,label = ad[70000]

#print(data.size(), s,label)

#loader = data_loader('./data/rock_bossanova_funk_RnB', batch_size=2)

#data,s,label = next(iter(loader))

#print(data.size(),label)

class TestSet(object):

    """docstring for TestSet."""

    def __init__(self, datadir:str):

        super(TestSet, self).__init__()

        self.datadir = datadir

        

        

    def choose(self):

        '''choose one speaker for test'''

        r = random.choice(styles)

        return r

    

    def test_data(self, src_style=None):

        '''choose one speaker for conversion'''

        if src_style:

            r_s = src_style

        else:

            r_s = self.choose()

        p = os.path.join(self.datadir, r_s)

        npyfiles = librosa.util.find_files(p, ext='npy')

       

        res = {}

        for f in npyfiles:

            filename = os.path.basename(f)

            mid = np.load(f)

            



            if not res.__contains__(filename):

                res[filename] = {}

            res[filename] = mid

            

        return res , r_s  

t = TestSet('data/test')
#print(t[0])
#d, style = t.test_data()



#for filename, content in d.items():
    #coded_mid = content
    
    #content_re = content.reshape(1, content.shape[1], content.shape[2],content.shape[0])
    #print(content_re.shape)





    #save_midis(content_re, './out_test/{}.mid'.format(filename))
    #content_re = content.reshape(1, content.shape[0], content.shape[1], 1)

    #print(content_re.shape)

    #     if  f_len >= FRAMES: 

    #         pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)

    #     elif f_len < FRAMES:

    #         pad_length = FRAMES - f_len

        

    #     coded_sp_norm = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))

    #     print('after:' , coded_sp_norm.shape)