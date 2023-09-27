import argparse
import time
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from models import *
from data_loader import loader
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='MvCLN in PyTorch')
parser.add_argument('--data', default='0', type=int,help='choice of dataset')
parser.add_argument('-li', '--log-interval', default='1', type=int, help='interval for logging info')
parser.add_argument('-bs', '--batch-size', default='128', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='10', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float, help='learning rate of adam')
parser.add_argument('-np', '--neg-prop', default='1', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-m', '--margin', default='1', type=int, help='initial margin')
parser.add_argument('--gpu', default='1', type=str, help='GPU device idx to use.')
parser.add_argument('--model_name', '-mn', type=str, default='LFA-CDFD.pkl')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []

class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, args):
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
        torch.clamp(torch.pow(pair_dist, 0.5) * (margin - pair_dist), min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss

def train(train_loader, train_dataset_len, best_train_acc, model, criterion, optimizer0, optimizer1, best_model_wts, epoch, args, output_path):
    pos_dist = 0  # mean distance of pos. pairs
    neg_dist = 0
    false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
    true_neg_dist = 0
    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0
    
    loss_list_train = []
    acc_list_train = []
    if epoch % args.log_interval == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))

    time0 = time.time()
    ncl_loss_value = 0
    ver_loss_value = 0
    cls_loss_value = 0
    iteration = 0 
    train_corrects = 0.0
    train_loss = 0.0

    model_name = args.model_name
    ncl_loss_list = []
    
    for batch_idx, (x0, x1, labels, real_labels, class_labels0, class_labels1) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        iter_corrects = 0.0
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels.to(device)
        class_labels0, class_labels1 = class_labels0.to(device), class_labels1.to(device)
        class_labels0 = class_labels0.long()
        class_labels1 = class_labels1.long()
        x0 = x0.view(x0.size()[0], -1)      #256
        x1 = x1.view(x1.size()[0], -1)
        
        f = (0 == class_labels0.data).sum().to(torch.float32)
        z = (1 == class_labels0.data).sum().to(torch.float32)
        h0, h1, z0, z1, y = model(x0, x1, f, z, class_labels0, args.batch_size)

        pair_dist = F.pairwise_distance(h0, h1)  # use Euclidean distance to measure similarity
        pos_dist += torch.sum(pair_dist[labels == 1])
        neg_dist += torch.sum(pair_dist[labels == 0])
        true_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_dist += torch.sum(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])
        pos_count += len(pair_dist[labels == 1])
        neg_count += len(pair_dist[labels == 0])
        true_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 0)])
        false_neg_count += len(pair_dist[torch.logical_and(labels == 0, real_labels == 1)])
        _, preds = torch.max(y.data, 1)
 
        ncl_loss = criterion[0](pair_dist, labels, args.margin, args)
        ver_loss = criterion[1](x0, z0) + criterion[1](x1, z1)
        cls_loss = criterion[2](y, class_labels0)
    
        loss = 0.01*ncl_loss + ver_loss + cls_loss
        ncl_loss_list.append(ncl_loss.item()) 
        
        ncl_loss_value += ncl_loss.item()
        ver_loss_value += ver_loss.item()
        cls_loss_value += cls_loss.item() 

        if epoch != 0:
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step() 
        
        iter_loss = cls_loss.data.item()
        train_loss += iter_loss
        iter_corrects = (preds == class_labels0.data).sum().to(torch.float32) 
        train_corrects += iter_corrects
        iteration += 1
        if not (iteration % 20):
            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / args.batch_size,
                                                                       iter_corrects / args.batch_size)) 
    print('*******')   
    cls_loss_value = 0
    iteration = 0 
    train_corrects = 0.0
    train_loss = 0.0
    for batch_idx, (x0, x1, labels, real_labels, class_labels0, class_labels1) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        iter_corrects = 0.0
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels.to(device)
        class_labels0, class_labels1 = class_labels0.to(device), class_labels1.to(device)
        class_labels0 = class_labels0.long()
        class_labels1 = class_labels1.long()
        x0 = x0.view(x0.size()[0], -1)      #256
        x1 = x1.view(x1.size()[0], -1)
        
        f = (0 == class_labels0.data).sum().to(torch.float32)
        z = (1 == class_labels0.data).sum().to(torch.float32)
        h0, h1, z0, z1, y = model(x0, x1, f, z, class_labels0, args.batch_size)
        _, preds = torch.max(y.data, 1)
        cls_loss = criterion[2](y, class_labels0)
        cls_loss_value += cls_loss.item() 

        if epoch != 0:
            optimizer1.zero_grad()
            cls_loss.backward()
            optimizer1.step()
        
        iter_loss = cls_loss.data.item()
        train_loss += iter_loss
        iter_corrects = (preds == class_labels0.data).sum().to(torch.float32) 
        train_corrects += iter_corrects
        iteration += 1
        if not (iteration % 20):
            print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / args.batch_size,
                                                                       iter_corrects / args.batch_size)) 
                
    epoch_time = time.time() - time0
    epoch_train_loss = train_loss / train_dataset_len
    epoch_train_acc = train_corrects / train_dataset_len
    print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
    loss_list_train.append(epoch_train_loss)
    acc_list_train.append(epoch_train_acc)
    if epoch_train_acc > best_train_acc:
        best_train_acc = epoch_train_acc
        print('epoch train best acc:Acc: {:.4f}'.format(best_train_acc))
        best_model_wts = model.state_dict()

    if not (epoch % 10):
        # Save the model trained with multiple gpu
        torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Model Best train Acc: {:.4f}'.format(best_train_acc))
   
    pos_dist /= pos_count
    neg_dist /= neg_count
    true_neg_dist /= true_neg_count
    false_neg_dist /= false_neg_count
    # margin = the pos. distance + neg. distance before training
    if epoch == 0 and args.margin != 1.0:
        args.margin = max(1, round((pos_dist + neg_dist).item()))
        logging.info("margin = {}".format(args.margin))

    if epoch % args.log_interval == 0:
        logging.info("dist: P = {}, N = {}, TN = {}, FN = {}; ncl_loss: {}, ver_loss:{}, time = {} s"
                     .format(round(pos_dist.item(), 2), round(neg_dist.item(), 2),
                             round(true_neg_dist.item(), 2), round(false_neg_dist.item(), 2),
                             round(ncl_loss_value / len(train_loader),2),
                             round(ver_loss_value / len(train_loader), 4), round(epoch_time, 2)))

    return pos_dist, neg_dist, false_neg_dist, true_neg_dist, epoch_time, best_model_wts, best_train_acc

def main():                                                                     # deep features of Caltech101
    data_name = ['FF++c23train','FF++c40train','NT23train','DP23train_lbp_hog','DP23train_mouth1to1','DP23train_global']
    NetSeed = 64
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)
    torch.cuda.manual_seed(NetSeed)
    train_pair_loader, train_dataset_len = loader(args.batch_size, args.neg_prop, data_name[args.data])
    name = 'CDFD'
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)       
    model = CDFD().cuda()
    
    criterion_ncl = NoiseRobustLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    params0 = list(model.encoder0.parameters()) + list(model.encoder1.parameters())+ list(model.decoder0.parameters())+list(model.decoder1.parameters())
    optimizer0 = torch.optim.Adam(params0, lr=args.learn_rate, weight_decay=0.0000000000001)
    params1 = list(model.classifier.parameters())
    optimizer1 = torch.optim.Adam(params1, lr=args.learn_rate, weight_decay=0.000000000000001)    

    if not os.path.exists('./log/'):
        os.mkdir("./log/")
        if not os.path.exists('./log/' + str(data_name[args.data]) + '/'):
            os.mkdir('./log/' + str(data_name[args.data]) + '/')
    path = os.path.join("./log/" + str(data_name[args.data]) + "/" + 'time=' + time
                        .strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path + '.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("******** Training begin ********")

    acc_list, nmi_list, ari_list = [], [], []
    train_time = 0
    best_train_acc = 0.0
    best_model_wts = model.state_dict() 
    for epoch in range(0, args.epochs + 1):
        if epoch == 0:
            with torch.no_grad():   
                pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time, best_model_wts, best_train_acc = \
                    train(train_pair_loader, train_dataset_len, best_train_acc, model,  [criterion_ncl, criterion_mse, criterion_cls], optimizer0, optimizer1, best_model_wts, epoch, args, output_path)
        else:
            pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, epoch_time, best_model_wts, best_train_acc = \
                train(train_pair_loader, train_dataset_len, best_train_acc, model,  [criterion_ncl, criterion_mse, criterion_cls], optimizer0, optimizer1, best_model_wts, epoch, args, output_path)
            
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))
        train_time += epoch_time
        pos_dist_mean_list.append(pos_dist_mean.item())
        neg_dist_mean_list.append(neg_dist_mean.item())
        true_neg_dist_mean_list.append(true_neg_dist_mean.item())
        false_neg_dist_mean_list.append(false_neg_dist_mean.item())

if __name__ == '__main__':
    main()
