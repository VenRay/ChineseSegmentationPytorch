import json
import pickle as pk

import re

import numpy as np

from gensim.corpora import Dictionary


embed_len = 200
min_freq = 3
max_vocab = 5000
seq_len = 50

pad_ind, oov_ind = 0, 1

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'


def convert(texts):
    sents, labels = list(), list()
    for text in texts:
        sent = re.sub(' ', '', text)
        sents.append(sent)
        inds, count = [0] * len(sent), 0
        for i in range(len(text)):
            if text[i] == ' ':
                count = count + 1
                inds[i - count] = 1
        labels.append(inds)
    return sents, labels


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def embed(sent_words, path_word_ind, path_word_vec, path_embed):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    #print (word_inds)
    #随机排布
    word_inds = tran_dict(word_inds, off=2)

    with open(path_word_ind, 'wb') as f:
        pk.dump(word_inds, f)
    #输出
    #print (word_inds)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    #print (word_vecs)
    vocab = word_vecs.vocab
    print (word_vecs['A'].shape)
    #200
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
                #嵌入规则为word_vecs

    print (embed_mat.shape)
    #(3571,200)
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def sent2ind(words, word_inds, seq_len, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    return pad(seq, seq_len, pad_ind)


def pad(seq, seq_len, val):
    if len(seq) < seq_len:
        return [val] * (seq_len - len(seq)) + seq
    else:
        return seq[-seq_len:]


def align(sent_words, labels, path_sent, path_label):
    with open(path_word_ind, 'rb') as f:
        word_inds = pk.load(f)
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, word_inds, seq_len, keep_oov=True)
        pad_seqs.append(pad_seq)
    pad_seqs = np.array(pad_seqs)
    print (pad_seqs[30])
    #(13638, 50)
    ind_mat = list()
    for label in labels:
        ind_mat.append(pad(label, seq_len, val=-1))
    ind_mat = np.array(ind_mat)
    print (ind_mat[0])
    #(13638, 50)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(ind_mat, f)


def vectorize(path_data, path_sent, path_label, mode):
    with open(path_data, 'r') as f:
        texts = json.load(f)
    sents, labels = convert(texts)
    #print (sents)
    #print (labels)
    #生成文字
    sent_words = [list(sent) for sent in sents]
    if mode == 'train':
        embed(sent_words, path_word_ind, path_word_vec, path_embed)
    align(sent_words, labels, path_sent, path_label)


if __name__ == '__main__':
    #70 20 10
    path_data = 'data/train.json'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_sent, path_label, 'train')
    path_data = 'data/dev.json'
    path_sent = 'feat/sent_dev.pkl'
    path_label = 'feat/label_dev.pkl'
    vectorize(path_data, path_sent, path_label, 'dev')
    path_data = 'data/test.json'
    path_sent = 'feat/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_sent, path_label, 'test')
