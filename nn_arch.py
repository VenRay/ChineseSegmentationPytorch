import torch.nn as nn
import torch

#architure of network
class lstm(nn.Module):
    def __init__(self, embed_mat, bidirect, layer_num):
        #embed_mat不影响输出维度
        super(lstm, self).__init__()
        #print ('a',embed_mat.shape)
        #torch.Size([3571, 200])
        vocab_num, embed_len = embed_mat.size()
        #print (vocab_num)
        #3571,200
        feat_len = 400 if bidirect else 200
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.lstm = nn.LSTM(embed_len, 200, batch_first=True,
                         bidirectional=bidirect, num_layers=layer_num)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(feat_len, 1))

    def forward(self, x):
        #print ('x',x.shape)
        #x.shape [128,50]
        x = self.embed(x)
        #print ('x',x.shape)
        #[128,50,200]
        h, h_n = self.lstm(x)
        #h,h_n = self.lstm(h)
        #print ('h',h.shape)
        #[128,50,1]
        return self.dl(h)

class cnn(nn.Module):
    def __init__(self,embed_mat):
        super(cnn,self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)

        self.Conv1d = nn.Conv1d(in_channels = 50,out_channels = 50,kernel_size=3,stride = 1,padding = 0)
        self.relu = nn.ReLU(inplace=True)

        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(196, 1))
    def forward(self,x):
        x = self.embed(x)

        x = self.relu(self.Conv1d(x))
        #print (x.shape)
        h = self.Conv1d(x)

        #print (x.shape)
        h = self.dl(h)

        return h
class gru(nn.Module):
    def __init__(self, embed_mat, bidirect, layer_num):
        super(gru, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        feat_len = 400 if bidirect else 200
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.gru = nn.GRU(embed_len, 200, batch_first=True,
                             bidirectional=bidirect, num_layers=layer_num)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(feat_len, 1))

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.gru(x)
        return self.dl(h1)

class lstm_cnn (nn.Module):
    def __init__(self, embed_mat, bidirect, layer_num):
        super(lstm_cnn, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        feat_len = 400 if bidirect else 200
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.lstm = nn.LSTM(embed_len, 200, batch_first=True,
                         bidirectional=bidirect, num_layers=layer_num)
        self.Conv1d = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(196, 1))

    def forward(self, x):
        #print ('x',x.shape)
        #x.shape [128,50]
        x = self.embed(x)

        h, h_n = self.lstm(x)
        h = self.relu(self.Conv1d(h))

        h = self.Conv1d(h)

        #[128,50,1]
        #print ('h',self.dl(h).shape)
        return self.dl(h)

class cnn_lstm (nn.Module):
    def __init__(self, embed_mat, bidirect, layer_num):
        super(cnn_lstm, self).__init__()
        vocab_num, embed_len = embed_mat.size()

        feat_len = 400 if bidirect else 200
        self.relu = nn.ReLU(inplace=True)
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)

        len = 196
        self.lstm = nn.LSTM(len,embed_len , batch_first=True,
                         bidirectional=bidirect, num_layers=layer_num)

        self.Conv1d = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(feat_len, 1))
    def forward(self, x):
        #print ('x',x.shape)
        #x.shape [128,50]
        x = self.embed(x)
        #print ('x',x.shape)
        x = self.Conv1d(x)
        h = self.relu(self.Conv1d(x))
        #print ('h',h.shape)
        h, h_n = self.lstm(h)
        #print (h.shape)
        #print (h.shape)
        #[128,50,1]
        #print ('h',self.dl(h).shape)
        return self.dl(h)

class paralle (nn.Module):
    def __init__(self, embed_mat, bidirect, layer_num):
        super(paralle, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.relu = nn.ReLU(inplace=True)

        feat_len = 400 if bidirect else 200

        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)

        len = 588
        self.lstm = nn.LSTM(len,embed_len , batch_first=True,
                         bidirectional=bidirect, num_layers=layer_num)

        self.Conv1d = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)

        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(embed_len, 1))

    def forward(self, x):
        x = self.embed(x)
        h1 = self.relu(self.Conv1d(x))
        x2 = self.Conv1d(x)
        #print (h1.shape)
        h2 = self.relu(self.Conv1d(x2))
        x3 = self.Conv1d(x2)
        #print (h2.shape)
        h3 = self.relu(self.Conv1d(x3))
        #print (h3.shape)
        h = torch.cat((h1,h2,h3),dim = -1)
        #print (h.shape)
        h,h_n = self.lstm(h)
        #print (h.shape)
        return self.dl(h)