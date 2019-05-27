import time

import pickle as pk

import torch
from torch.nn import BCEWithLogitsLoss,BCELoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from nn_arch import lstm, gru,cnn,lstm_cnn,cnn_lstm,paralle

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#cuda
detail = False if torch.cuda.is_available() else True

batch_size = 128

#the path of word embedding
path_embed = 'feat/embed.pkl'

models = {'gru': gru,
         'cnn': cnn,
         'lstm_cnn': lstm_cnn,
         'lstm':lstm,
         'cnn_lstm':cnn_lstm,
          'paralle':paralle
         }


with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

#path of models
paths = {
         'cnn':'weight/cnn.pkl',
         'lstm':'weight/lstm.pkl',
         'cnn_lstm':'weight/cnn_lstm.pkl',
         'lstm_cnn':'weight/lstm_cnn.pkl',
         'paralle':'weight/paralle.pkl'
         }

#load data
def load_feat(path_feats):
    with open(path_feats['sent_train'], 'rb') as f:
        train_sents = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['sent_dev'], 'rb') as f:
        dev_sents = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_sents, train_labels, dev_sents, dev_labels


def step_print(step, batch_loss, batch_acc):
    print('\n{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step, batch_loss, batch_acc))


def epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra):
    print('\n{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
          'epoch', epoch, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def tensorize(feats, device):
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat).to(device))
    return tensors

#dataloader
def get_loader(pairs):
    sents, labels = pairs
    pairs = TensorDataset(sents, labels)
    return DataLoader(pairs, batch_size, shuffle=True)


def get_metric(model, loss_func, pairs, thre):
    sents, labels = pairs
    #torch.Size([128, 50])
    prods = model(sents)
    #torch.Size([128, 50, 1])

    prods = torch.squeeze(prods, dim=-1)
    mask = labels > -1
    mask_prods, mask_labels = prods.masked_select(mask), labels.masked_select(mask)
    mask_preds = mask_prods > thre

    loss = loss_func(mask_prods, mask_labels.float())
    acc = (mask_preds == mask_labels.byte()).sum().item()
    return loss, acc, len(mask_preds)

#train
def batch_train(model, loss_func, optim, loader, detail):
    total_loss, total_acc, total_num = [0] * 3
    for step, pairs in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, pairs, thre=0.5)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
        if detail:
            step_print(step + 1, batch_loss / batch_num, batch_acc / batch_num)
    return total_loss / total_num, total_acc / total_num

#assess by development data
def batch_dev(model, loss_func, loader):
    total_loss, total_acc, total_num = [0] * 3
    for step, pairs in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, pairs, thre=0.5)
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
    return total_loss / total_num, total_acc / total_num

#main
def fit(name, max_epoch, embed_mat, path_feats, detail):


    tensors = tensorize(load_feat(path_feats), device)

    bound = int(len(tensors) / 2)
    #divide data and label
    train_loader, dev_loader = get_loader(tensors[:bound]), get_loader(tensors[bound:])

    embed_mat = torch.Tensor(embed_mat)
    # (embed_mat.shape) [3571, 200]

    bidirect = False
    if name == 'cnn':
        model = models[name](embed_mat).to(device)
    else:
        model = models[name](embed_mat, bidirect, layer_num=1).to(device)

    #summary(model,(128,50))
    loss_func = BCEWithLogitsLoss(reduction='sum')

    learn_rate, min_rate = 1e-3, 1e-5
    min_dev_loss = float('inf')
    trap_count, max_count = 0, 3
    print('\n{}'.format(model))
    train, epoch = True, 0
    loss = []
    d_acc = []
    t_acc = []
    while train and epoch < max_epoch:
        epoch = epoch + 1
        model.train()
        optim = Adam(model.parameters(), lr=learn_rate)

        start = time.time()
        train_loss, train_acc = batch_train(model, loss_func, optim, train_loader, detail)

        delta = time.time() - start
        #estimate by dev
        with torch.no_grad():
            model.eval()
            dev_loss, dev_acc = batch_dev(model, loss_func, dev_loader)
        extra = ''
        loss.append(dev_loss)
        d_acc.append(dev_acc)
        t_acc.append(train_acc)
        if dev_loss < min_dev_loss:
            extra = ', val_loss reduce by {:.3f}'.format(min_dev_loss - dev_loss)
            min_dev_loss = dev_loss
            trap_count = 0
            torch.save(model, paths[name])

        else:
            trap_count = trap_count + 1
            if trap_count > max_count:
                learn_rate = learn_rate / 10

        epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra)
    #write acc and loss to excel
    data_acc = pd.DataFrame(t_acc)
    data_loss = pd.DataFrame(d_acc)

    writer = pd.ExcelWriter('./acc/{}.xlsx'.format(name))
    data_acc.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
    data_loss.to_excel(writer, 'page_2', float_format='%.5f')
    writer.save()



if __name__ == '__main__':
    path_feats = dict()
    path_feats['sent_train'] = 'feat/sent_train.pkl'
    path_feats['label_train'] = 'feat/label_train.pkl'
    path_feats['sent_dev'] = 'feat/sent_dev.pkl'
    path_feats['label_dev'] = 'feat/label_dev.pkl'
    #fit('lstm', 20, embed_mat, path_feats, detail)
    #fit('cnn',20,embed_mat, path_feats, detail)
    fit('lstm_cnn', 20, embed_mat, path_feats, detail)
    #fit('cnn_lstm', 20, embed_mat, path_feats, detail)
    #fit('paralle',20, embed_mat, path_feats, detail)