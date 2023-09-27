import torch.nn as nn
import torch
import numpy as np
import struct
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CDFD(nn.Module):  # 256,2304
    def __init__(self,num_classes=2):
        super(CDFD, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 256))
        self.decoder1 = nn.Sequential(nn.Linear(20, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 2304))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=20, out_features=num_classes)
        )
    def forward(self, x0, x1, f, z, class_labels0, batchsize):
        h0 = self.encoder0(x0)     #1024,10
        h1 = self.encoder1(x1)
        union = torch.cat([h0, h1], 1)    #1024,20

        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        
        w = torch.zeros((batchsize,batchsize)) 
        for i in range(batchsize):
            for j in range(batchsize):
                if i == j:
                    if class_labels0[i] == 1:
                        w[i,j] = 2*(f/batchsize)
                    else:
                        w[i,j] = z/batchsize
        union = union.cpu()
        union = torch.matmul(w, union)
        union = union.to(device)
        
        y = self.classifier(union)
        y = F.softmax(y, dim=1)
        return h0, h1, z0, z1, y 