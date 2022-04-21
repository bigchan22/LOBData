#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import load
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse

from timeit import default_timer as timer
from tensorboardX import SummaryWriter
from datetime import datetime
from source.Data_load import Data_load
from source.transformer import Seq2SeqTransformer
from source.train import train_epoch,evaluate

parser = argparse.ArgumentParser(description='Train Config')
# In[2]:
parser.add_argument('--epoch',          type=int,   default=3000)
parser.add_argument('--batch_size',     type=int,   default=128)
parser.add_argument('--lr_initial',     type=float, default=1e-3)
parser.add_argument('--hid_dim',     type=int, default=128)
parser.add_argument('--emb_dim',     type=int, default=128)
parser.add_argument('--num_layers',     type=int, default=4)
parser.add_argument('--num_head',     type=int, default=8)

args= parser.parse_args()

print(args.epoch)
print(args.batch_size)
print(args.lr_initial)
print(args.emb_dim)
print(args.hid_dim)
print(args.num_head)
print(args.num_layers)

num_epochs=args.epoch
bptt = 39
TGT_VOCAB_SIZE = 3
EMB_SIZE = args.emb_dim
NHEAD = args.num_head
FFN_HID_DIM = args.hid_dim
BATCH_SIZE = args.batch_size
lr_init=args.lr_initial
NUM_ENCODER_LAYERS = (args.num_layers)//2
NUM_DECODER_LAYERS = (args.num_layers)//2


# load array
XData = load('./TrainData/Train_Mid_2022_4_12_19_34.npy')
YData = load('./TrainData/Train_Mid_Label_2022_4_12_19_34.npy')
Xdata=[]
Ydata=[]
Xtrain_data=[]
Ytrain_data=[]
Xtest_data=[]
Ytest_data=[]



for idx in range(len(XData)//39):
    if(np.isinf(XData[39*idx:39*(idx+1)][:,:].tolist()).any()):
        print(np.isinf(XData[39*idx:39*(idx+1)][:,:].tolist()).any())
        continue
    if(np.isinf(YData[39*idx:39*(idx+1)][:].tolist()).any()):
        print(np.isinf(YData[39*idx:39*(idx+1)].tolist()).any())
        continue
    if idx< (len(XData)//39)*0.9:
        Xtrain_data.append(XData[39*idx:39*(idx+1)][:,:-1].tolist())
        Ytrain_data.append(YData[39*idx:39*(idx+1)].tolist())
    else:
        Xtest_data.append(XData[39*idx:39*(idx+1)][:,:-1].tolist())
        Ytest_data.append(YData[39*idx:39*(idx+1)].tolist())
Xtrain_data=np.vstack(Xtrain_data)
Ytrain_data=np.vstack(Ytrain_data)
Xtest_data=np.vstack(Xtest_data)
Ytest_data=np.vstack(Ytest_data)

Xtrain_data=torch.FloatTensor(Xtrain_data)
Xtest_data=torch.FloatTensor(Xtest_data)
Ytrain_data=torch.LongTensor(Ytrain_data)
Ytest_data=torch.LongTensor(Ytest_data)

Ytrain_data=Ytrain_data.view(-1)
Ytest_data=Ytest_data.view(-1)






torch.manual_seed(0)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SRC_VOCAB_SIZE = Xtrain_data.shape[1]


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
summary = SummaryWriter()
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(transformer.parameters(), lr=lr_init, betas=(0.9, 0.98), eps=1e-9)
Val_loss=[]
Train_loss=[]
Accuracy=[]
F1score=[]
NUM_EPOCHS = num_epochs
best_val_loss=100000000
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss,_ = train_epoch(transformer, optimizer,Xtrain_data,Ytrain_data,loss_fn,device,BATCH_SIZE,bptt)
    end_time = timer()
    val_loss,acc,prec,reca,f1sc,confusion = evaluate(transformer,Xtest_data,Ytest_data,loss_fn,device,BATCH_SIZE,bptt)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_confusion=confusion
        best_acc=acc
        best_prec=prec
        best_reca=reca
        best_f1sc=f1sc
        best_model = transformer
    Val_loss.append(val_loss)
    Train_loss.append(train_loss)
    Accuracy.append(acc)
    F1score.append(f1sc)
now = datetime.now()
now.strftime("%m/%d/%Y, %H:%M:%S")

date_time = now.strftime("%m_%d_%Y")


# In[ ]:


PATH='results/best_model_Trans_seq_'+date_time+'_midprice'

torch.save(best_model.state_dict(), PATH)

file_name='results/result_Trans_Mid'+date_time+'.txt'
text_to_append=PATH+'\t'+"Acc:"+str(best_acc)+'\t'+"prec:"+str(best_prec)+'\t'+"recall:"+str(best_reca)+'\t'+"f1sc:"+str(best_f1sc)
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


# In[ ]:



with open("results/Val_loss_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Val_loss, fp)
    
with open("results/Train_loss_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Train_loss, fp)
    
with open("results/Accuracy_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Accuracy, fp)
    
with open("results/F1_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(F1score, fp)
