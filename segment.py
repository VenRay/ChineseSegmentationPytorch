import pickle as pk

import numpy as np

import torch

from represent import sent2ind


device = torch.device('cpu')

seq_len = 50

path_word_ind = 'feat/word_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)


paths = {'gru': 'weight/gru.pkl',
         'cnn':'weight/cnn.pkl',
         'lstm':'weight/lstm.pkl',
         'cnn_lstm':'weight/cnn_lstm.pkl',
         'lstm_cnn':'weight/lstm_cnn.pkl',
         'paralle':'weight/paralle.pkl'
         }

models = {'lstm': torch.load(paths['lstm'], map_location=device),
          'cnn': torch.load(paths['cnn'], map_location=device),
          'lstm_cnn': torch.load(paths['lstm_cnn'], map_location=device),
          'cnn_lstm': torch.load(paths['cnn_lstm'], map_location=device),
          'paralle':torch.load(paths['paralle'], map_location=device)
            }


def predict(text, name, thre):
    text = text.strip()
    pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
    model = models[name]
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sent))
    probs = probs.numpy()[0]
    probs = np.squeeze(probs, axis=-1)
    preds = probs > thre
    bound = min(len(text), seq_len)
    mask_preds = preds[-bound:]
    cands = list()
    for word, pred in zip(text, mask_preds):
        cands.append(word)
        if pred:
            cands.append(' ')
    return ''.join(cands)


if __name__ == '__main__':

        #text = input('text: ')
    text = '我的烤面筋融化你的心'
    print('lstm_cnn: %s' % predict(text, 'lstm_cnn', thre=0.5))
        #print('rnn_bi: %s' % predict(text, 'rnn_bi', thre=0.5))
        #print('s2s: %s' % predict(text, 's2s', thre=0.5))
        #print('s2s_bi: %s' % predict(text, 's2s_bi', thre=0.5))
