import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import TT_split, normalize
import torch
import random
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder

def load_data(dataset, neg_prop):
    all_data = []
    train_pairs = []
    label = []
    g_label = []
    l_label = []

    mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'DP23train_lbp_hog':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 6941
        
    if dataset == 'FF++c23train':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 7000
        
    if dataset == 'FF++c40train':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 7000
        
    if dataset == 'FF++c23val':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 3000
        
    if dataset == 'FF++c40val':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 3000
        
    if dataset == 'DFDCval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 637
        
    if dataset == 'NT23train':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 7000   
        
    if dataset == 'DP40lbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2969
        
    if dataset == 'F240lbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2996
        
    if dataset == 'Celeb-DFv2val':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 847
        
    if dataset == 'FS40lbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2996
        
    if dataset == 'deepfakelbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2845
        
    if dataset == 'NT23lbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2996
        
    if dataset == 'FS23val':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2996
        
    if dataset == 'Celeb_DF_V2_lbphogval':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 1435
        
    if dataset == 'DP23train_mouth':
        data = mat['data']  # 
        label1 = mat['labels1']
        label2 = mat['labels2']
        num = 6941
        
    if dataset == 'DP23val_lbp_hog':
        lbp = mat['lbp_features']  # 
        hog = mat['hog_features']
        labels = mat['labels']
        num = 2951
        
    if dataset == 'DP23val_mouth':
        data = mat['data']  # 
        label1 = mat['labels1']
        label2 = mat['labels2'] 
        num = 2930     
        
    end = num*2
    g_idx = list(range(end))  
    l_idx = list(range(end))  

    for i in range(len(g_idx)):
        if g_idx[i] < num:
            g_label.append(0)
        else:
            g_label.append(1)
    for j in range(len(l_idx)):
        if l_idx[j] < num:
            l_label.append(0)
        else:
            l_label.append(1)      
    train_X, train_Y = lbp[g_idx], hog[l_idx]

    if dataset == 'DFDCval':
        view0, view1, noisy_labels, real_labels, class_labels0, class_labels1 = get_pairs_val(train_X, train_Y, g_label, l_label)
    else:
        view0, view1, noisy_labels, real_labels, class_labels0, class_labels1 = get_pairs(train_X, train_Y, neg_prop, g_label, l_label)
    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1

    train_pair_labels = real_labels
    train_pairs.append(view0)
    train_pairs.append(view1)
    train_pair_real_labels = real_labels

    return train_pairs, train_pair_labels, train_pair_real_labels, class_labels0, class_labels1

def get_pairs(train_X, train_Y, neg_prop, g_label, l_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs   
    for i in range(len(train_X)):
        #if g_label[i] == 0:
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(g_label[i])
        class_labels1.append(l_label[i])
    
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)
        for k in range(neg_prop):
            if g_label[j]!=1 or l_label[neg_idx[k]]!=1:
                view0.append(train_X[j])
                view1.append(train_Y[neg_idx[k]])
                labels.append(0)
                class_labels0.append(0)
                class_labels1.append(0)
                real_labels.append(0)
    
    labels = np.array(labels, dtype=np.int16)
    real_labels = np.array(real_labels, dtype=np.int16)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int16), np.array(class_labels1, dtype=np.int16)
    view0, view1 = np.array(view0, dtype=np.float16), np.array(view1, dtype=np.float16)
    return view0, view1, labels, real_labels, class_labels0, class_labels1
def get_pairs_val(train_X, train_Y, g_label, l_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []

    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(g_label[i])
        class_labels1.append(l_label[i])

    labels = np.array(labels, dtype=np.int16)
    real_labels = np.array(real_labels, dtype=np.int16)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int16), np.array(class_labels1, dtype=np.int16)
    view0, view1 = np.array(view0, dtype=np.float16), np.array(view1, dtype=np.float16)
    return view0, view1, labels, real_labels, class_labels0, class_labels1

class getDataset(Dataset):
    def __init__(self, data, labels, real_labels, class_labels0, class_labels1):
        self.data = data 
        self.labels = labels
        self.real_labels = real_labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        
    def __getitem__(self, index):
        #index = index % 256
        fea0, fea1 = (torch.from_numpy(self.data[0][index, :])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][index, : ])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int16(self.labels[index])
        real_label = np.int16(self.real_labels[index])
        class_label0 = np.int16(self.class_labels0[index])
        class_label1 = np.int16(self.class_labels1[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        return fea0, fea1, label, real_label, class_label0, class_label1

    def __len__(self):
        return len(self.labels) 
    
def loader(train_bs, neg_prop, dataset):
    """
    :param train_bs: batch size for training, default is 128
    :param neg_prop: negative / positive pairs' ratio
    :param dataset: choice of dataset
    """
    train_pairs, train_pair_labels, train_pair_real_labels, class_labels0, class_labels1 \
     = load_data(dataset, neg_prop)
    train_pair_dataset = getDataset(train_pairs, train_pair_labels, train_pair_real_labels, class_labels0, class_labels1)
    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    train_dataset_len = len(train_pair_dataset)
    return train_pair_loader,train_dataset_len
