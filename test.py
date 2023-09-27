import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from torch.utils.data import DataLoader
import argparse
from models import *
import torchvision
from PIL import Image
from data_loader import loader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def main():
    data_name = ['DFDCval']
    args = parse.parse_args()
    batch_size = args.batch_size
    model_path = args.model_path
    torch.backends.cudnn.benchmark = True

    train_loader, train_dataset_len = loader(args.batch_size, args.neg_prop, data_name[0])

    acc = 0
    corrects_sum = 0
    all_y = []
    all_labels = []
    model = CDFD().to(device)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x0, x1, labels, real_labels, class_labels0, class_labels1) in enumerate(train_loader):
            iter_corrects = 0.0
            corrects = 0
            corrects0 = 0
            corrects1 = 0
            x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels.to(device)
            class_labels0, class_labels1 = class_labels0.to(device), class_labels1.to(device)
            class_labels0 = class_labels0.long()
            class_labels1 = class_labels1.long()
            x0 = x0.view(x0.size()[0], -1)      #256
            x1 = x1.view(x1.size()[0], -1)

            f = (0 == class_labels0.data).sum().to(torch.float32)
            z = (1 == class_labels0.data).sum().to(torch.float32)
            try:
                h0, h1, z0, z1, y = model(x0, x1, f, z, class_labels0, batch_size)
            except:
                print("error raise in batch", batch_idx)

            _, preds = torch.max(y.data, 1)
            corrects = torch.sum(preds == class_labels0.data).to(torch.float32)
            corrects_sum += corrects
 
            y_g=y[:,1]
            all_y.append(y_g) 
            all_labels.append(class_labels0)
 
            print('Iteration Acc {:.4f}'.format(corrects / batch_size))

        acc = corrects_sum / train_dataset_len
        print('Test Acc: {:.4f}'.format(acc))
        print('Test corr: {:.4f}'.format(corrects))
        print('Test size: {:.4f}'.format(train_dataset_len))
  
        y_pred = torch.cat(all_y)
        y_true = torch.cat(all_labels) 
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()

        auc = roc_auc_score(y_true, y_pred)
        print("Overall AUC: ", auc)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse.add_argument('--batch_size', '-bz', type=int, default=128)

    #parse.add_argument('--test_path', '-tp', type=str, default='./datasets/val')

    parse.add_argument('--model_path', '-mp', type=str, default='./output/CDFD/best.pkl')
    
    parse.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs')

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    main()