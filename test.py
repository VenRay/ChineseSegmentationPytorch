import pickle as pk

import torch

from sklearn.metrics import f1_score, accuracy_score

from build import tensorize

from segment import models




device = torch.device('cpu')

seq_len = 50

path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'

with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sents, labels, thre):
    sents, labels = tensorize([sents, labels], device)
    model = models[name]
    #labels.shape 1949  50
    print (name)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sents))

    probs = torch.squeeze(probs, dim=-1)
    #print (probs.shape) torch.Size([1949, 50])
    mask = labels > -1
    #print (mask[0])
    mask_probs, mask_labels = probs.masked_select(mask), labels.masked_select(mask)
    #mask_select会将满足mask的指示，将mask == 1(实际文字）的点选出来。
    #print (mask_probs.shape) [71592]
    #print (mask_labels)
    #print (mask_probs)
    mask_preds = mask_probs > thre
    #实际预测
    print('\n%s f1: %.5f - acc: %.5f' % (name, f1_score(mask_labels, mask_preds),
                                         accuracy_score(mask_labels, mask_preds)))


if __name__ == '__main__':

    #test ('lstm', sents, labels, thre=0.5)
    #test ('cnn', sents, labels, thre=0.5)
    test('lstm_cnn',sents, labels, thre=0.5)
    test('cnn_lstm', sents, labels, thre=0.5)
    #test('paralle',sents, labels, thre=0.5)
    #test('s2s', sents, labels, thre=0.5)
    #test('rnn_bi', sents, labels, thre=0.5)
    #test('s2s', sents, labels, thre=0.5)
    #test('s2s_bi', sents, labels, thre=0.5)
