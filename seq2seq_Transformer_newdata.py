#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from numpy import load
from torch.utils.data import Dataset
from datetime import datetime
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from timeit import default_timer as timer
import time

from tensorboardX import SummaryWriter

import math

# In[2]:


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.src_tok_emb = nn.Linear(src_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                memory_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg).squeeze())
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask=memory_mask)
        return self.generator(outs)


# In[3]:


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_square_subsequent_mask2(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask


def generate_square_subsequent_mask3(sz):
    mask = (torch.triu(torch.ones((sz, sz - 1), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_mask = generate_square_subsequent_mask(src_seq_len)
    mmr_mask = generate_square_subsequent_mask3(src_seq_len)
    return src_mask, tgt_mask, mmr_mask


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz, bptt):
    # 데이터셋을 bsz 파트들로 나눕니다.
    if (data.size(0) % (bsz * bptt) != 0):
        data = data.view(-1, bptt, data.size(1)).transpose(0, 1).contiguous()
        return data.to(device)
    nbatch = data.size(0) // bsz
    # 깔끔하게 나누어 떨어지지 않는 추가적인 부분(나머지들) 은 잘라냅니다.
    data = data.narrow(0, 0, nbatch * bsz)
    # 데이터에 대하여 bsz 배치들로 동등하게 나눕니다.
    data = data.view(bsz, -1, data.size(1)).transpose(0, 1).contiguous()
    return data.to(device)


bptt = 39


def get_batch(source, i, bs):
    seq_len = min(bptt * bs, len(source) - i)
    data = source[i:i + seq_len]
    if (seq_len != bptt * bs):
        print(seq_len)
    target = source[i:i + seq_len].reshape(-1)
    return data, target


# In[5]:


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0

    i = 0
    while (i < Xtest_data.size(0)):
        data, org_targets = get_batch(Xtrain_data, i, BATCH_SIZE)
        if (data.isnan().any() or data.isinf().any()):
            print(data)
            continue
        targets, _ = get_batch(Ytrain_data, i, BATCH_SIZE)

        targets = torch.unsqueeze(targets, 1)
        src = batchify(data, BATCH_SIZE, bptt)
        tgt = batchify(targets, BATCH_SIZE, bptt)
        src_input = src[:]
        tgt_input = tgt[:-1]

        src_mask, tgt_mask, mmr_mask = create_mask(src_input, tgt_input)
        logits = model(src_input, tgt_input, src_mask, tgt_mask, mmr_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if (loss.isnan()):
            print('NNNNN')
            print(loss)
        _, predicted = torch.max(logits, -1)
        correct += (tgt_out.squeeze() == predicted).sum().item()
        total += len(predicted) * BATCH_SIZE
        tot0 += (0 == tgt_out.squeeze()).sum().item()
        tot1 += (1 == tgt_out.squeeze()).sum().item()
        tot2 += (2 == tgt_out.squeeze()).sum().item()
        correct0 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        correct1 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        correct2 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

        conf00 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf01 += ((0 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf02 += ((0 == predicted) & (2 == tgt_out.squeeze())).sum().item()
        conf10 += ((1 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf11 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf12 += ((1 == predicted) & (2 == tgt_out.squeeze())).sum().item()
        conf20 += ((2 == predicted) & (0 == tgt_out.squeeze())).sum().item()
        conf21 += ((2 == predicted) & (1 == tgt_out.squeeze())).sum().item()
        conf22 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

        i += targets.size()[0]
    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    f1sc = 2 * (prec * reca) / (prec + reca)
    print("Total:", total)
    print("Correct", correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtrain_data.size(0), [conf00, conf01, conf02, conf10, conf11, conf12, conf20, conf21, conf22]


# In[6]:


def evaluate(model):
    model.eval()
    losses = 0
    total = 0
    correct = 0
    correct1 = 0
    tot1 = 0
    correct2 = 0
    tot2 = 0
    correct0 = 0
    tot0 = 0
    conf00 = 0
    conf01 = 0
    conf02 = 0
    conf10 = 0
    conf11 = 0
    conf12 = 0
    conf20 = 0
    conf21 = 0
    conf22 = 0
    stime = time.time()
    with torch.no_grad():
        i = 0
        while (i < Xtest_data.size(0)):
            #         for i in range(0, Xtest_data.size(0) - 1, bptt):

            data, org_targets = get_batch(Xtest_data, i, BATCH_SIZE)
            if (data.isnan().any() or data.isinf().any()):
                continue
            targets, _ = get_batch(Ytest_data, i, BATCH_SIZE)
            targets = torch.unsqueeze(targets, 1)
            src = batchify(data, BATCH_SIZE, bptt)
            tgt = batchify(targets, BATCH_SIZE, bptt)

            src_input = src[:]

            tgt_input = tgt[:-1]
            #             print(src_input.size())
            #             print(tgt_input.size())
            src_mask, tgt_mask, mmr_mask = create_mask(src_input, tgt_input)

            #             logits = model(src_input, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            logits = model(src_input, tgt_input, src_mask, tgt_mask, mmr_mask)

            tgt_out = tgt[1:]

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if (loss.isnan()):
                print(src, tgt_input)
                break
            losses += loss.item()
            _, predicted = torch.max(logits, -1)

            correct += (tgt_out.squeeze() == predicted).sum().item()
            total += len(predicted) * BATCH_SIZE
            tot0 += (0 == tgt_out.squeeze()).sum().item()
            tot1 += (1 == tgt_out.squeeze()).sum().item()
            tot2 += (2 == tgt_out.squeeze()).sum().item()

            conf00 += ((0 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf01 += ((0 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf02 += ((0 == predicted) & (2 == tgt_out.squeeze())).sum().item()
            conf10 += ((1 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf11 += ((1 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf12 += ((1 == predicted) & (2 == tgt_out.squeeze())).sum().item()
            conf20 += ((2 == predicted) & (0 == tgt_out.squeeze())).sum().item()
            conf21 += ((2 == predicted) & (1 == tgt_out.squeeze())).sum().item()
            conf22 += ((2 == predicted) & (2 == tgt_out.squeeze())).sum().item()

            i += targets.size()[0]
    etime = time.time()
    print("Time elapsed", etime - stime)
    tp0 = conf00
    fp0 = conf01 + conf02
    fn0 = conf10 + conf20
    if (tp0 + fp0 == 0):
        prec0 = 0
    else:
        prec0 = tp0 / (tp0 + fp0)
    if (tp0 + fn0 == 0):
        reca0 = 0
    else:
        reca0 = tp0 / (tp0 + fn0)

    tp1 = conf11
    fp1 = conf10 + conf12
    fn1 = conf01 + conf21

    if (tp1 + fp1 == 0):
        prec1 = 0
    else:
        prec1 = tp1 / (tp1 + fp1)
    if (tp1 + fn1 == 0):
        reca1 = 0
    else:
        reca1 = tp1 / (tp1 + fn1)

    tp2 = conf22
    fp2 = conf20 + conf21
    fn2 = conf02 + conf12

    if (tp2 + fp2 == 0):
        prec2 = 0
    else:
        prec2 = tp2 / (tp2 + fp2)
    if (tp2 + fn2 == 0):
        reca2 = 0
    else:
        reca2 = tp2 / (tp2 + fn2)

    prec = (prec0 + prec1 + prec2) / 3
    reca = (reca0 + reca1 + reca2) / 3
    f1sc = 2 * (prec * reca) / (prec + reca)
    print(total)
    print(correct)
    print("Acc:", correct / total)
    print("Prec", prec)
    print("Recall", reca)
    print("F1", f1sc)
    return losses / Xtest_data.size(0), correct / total, prec, reca, f1sc, [conf00, conf01, conf02, conf10, conf11,
                                                                            conf12, conf20, conf21, conf22]


# In[7]:


mbrnlist1 = [(5, 194), (2, 155), (12, 100), (17, 29), (42, 1), (44, 1), (50, 92), (2, 83), (4, 10118), (8, 298)]
mbrnlist2 = [(5, 194), (12, 100), (2, 155), (17, 29), (42, 1), (44, 1), (2, 83), (4, 10118), (4, 9997), (50, 91)]

mbrnlist = mbrnlist1 + mbrnlist2
mbrnlist = set(mbrnlist)
mbrnlist = list(mbrnlist)

# load array
MBR_NO, BRN_NO = mbrnlist[2]
featnorm = True

# In[8]:


for MBR_NO, BRN_NO in mbrnlist:
    #     if featnorm==True:
    #         Data_train = load('DataMid_'+str(MBR_NO)+'_'+str(BRN_NO)+'norm_train.npy',allow_pickle=True)
    #         Data_test =  load('DataMid_'+str(MBR_NO)+'_'+str(BRN_NO)+'norm_test.npy',allow_pickle=True)
    #     else:
    #         Data_train = load('DataMid_'+str(MBR_NO)+'_'+str(BRN_NO)+'train.npy',allow_pickle=True)
    #         Data_test =  load('DataMid_'+str(MBR_NO)+'_'+str(BRN_NO)+'test.npy',allow_pickle=True)

    if featnorm == True:
        Data_train = load('OldData/Data0930_' + str(MBR_NO) + '_' + str(BRN_NO) + 'norm_train.npy', allow_pickle=True)
        Data_test = load('OldData/Data0930_' + str(MBR_NO) + '_' + str(BRN_NO) + 'norm_test.npy', allow_pickle=True)
    else:
        Data_train = load('OldData/Data0930_' + str(MBR_NO) + '_' + str(BRN_NO) + 'train.npy', allow_pickle=True)
        Data_test = load('OldData/Data0930_' + str(MBR_NO) + '_' + str(BRN_NO) + 'test.npy', allow_pickle=True)

    Xdata = []
    Ydata = []
    Xtrain_data = []
    Ytrain_data = []
    Xtest_data = []
    Ytest_data = []

    for idx in range(len(Data_train) // 39):
        if (np.isinf(Data_train[39 * idx:39 * (idx + 1)][:, :].tolist()).any()):
            print(np.isinf(Data_train[39 * idx:39 * (idx + 1)][:, :].tolist()).any())
            continue
        Xtrain_data.append(Data_train[39 * idx:39 * (idx + 1)][:, :].tolist())
        Ytrain_data.append(Data_train[39 * idx:39 * (idx + 1)][:, -1].tolist())
    for idx in range(len(Data_test) // 39):
        if (np.isinf(Data_test[39 * idx:39 * (idx + 1)][:, :].tolist()).any()):
            print(np.isinf(Data_test[39 * idx:39 * (idx + 1)][:, :].tolist()).any())
            continue
        Xtest_data.append(Data_test[39 * idx:39 * (idx + 1)][:, :].tolist())
        Ytest_data.append(Data_test[39 * idx:39 * (idx + 1)][:, -1].tolist())

    Xtrain_data = np.vstack(Xtrain_data)
    Ytrain_data = np.vstack(Ytrain_data)

    Xtrain_data = torch.FloatTensor(Xtrain_data)
    Ytrain_data = torch.IntTensor(Ytrain_data)
    Ytrain_data = Ytrain_data.view(-1)

    Xtest_data = np.vstack(Xtest_data)
    Ytest_data = np.vstack(Ytest_data)
    Xtest_data = torch.FloatTensor(Xtest_data)
    Ytest_data = torch.IntTensor(Ytest_data)
    Ytest_data = Ytest_data.view(-1)

    Ytrain_data = 2 * (Ytrain_data > 0).long() + (Ytrain_data == 0).long()
    Ytest_data = 2 * (Ytest_data > 0).long() + (Ytest_data == 0).long()
    Ytrain_data = Ytrain_data.T
    Ytest_data = Ytest_data.T

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = Xtrain_data.shape[1]
    TGT_VOCAB_SIZE = 3
    EMB_SIZE = 128
    NHEAD = 8
    FFN_HID_DIM = 128
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    summary = SummaryWriter()
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    NUM_EPOCHS = 1000
    best_val_loss = 100000000
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss, _ = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss, acc, prec, reca, f1sc, confusion = evaluate(transformer)
        print((
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_confusion = confusion
            best_acc = acc
            best_prec = prec
            best_reca = reca
            best_f1sc = f1sc
            best_model = transformer
        summary.add_scalar('val loss', val_loss, epoch)
        summary.add_scalar('total loss', train_loss, epoch)
        summary.add_scalar('f1 score', f1sc, epoch)
        summary.add_scalar('Accuracy', acc, epoch)
    now = datetime.now()
    now.strftime("%m/%d/%Y, %H:%M:%S")

    date_time = now.strftime("%m_%d_%Y")

    PATH = 'best_model_Trans_seq_' + date_time + '_' + str(MBR_NO) + '_' + str(BRN_NO)
    if featnorm == True:
        torch.save(best_model.state_dict(), PATH + 'norm')
    else:
        torch.save(best_model.state_dict(), PATH)
    if featnorm == True:
        file_name = 'result_Trans_' + date_time + '_norm.txt'
    else:
        file_name = 'result_Trans_' + date_time + '.txt'
    text_to_append = PATH + '\t' + "Acc:" + str(best_acc) + '\t' + "prec:" + str(best_prec) + '\t' + "recall:" + str(
        best_reca) + '\t' + "f1sc:" + str(best_f1sc)
    print(text_to_append)
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
