#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import load
import torch
import torch.nn as nn
import torch.nn.functional as F

from timeit import default_timer as timer


from tensorboardX import SummaryWriter
from datetime import datetime
from source.Data_load import Data_load
from source.transformer import Seq2SeqTransformer
from source.train import train_epoch,evaluate


# In[2]:


# load array
Data = load('./TrainData/GDF2022_3_27_16_52.csvTrain.npy')

Xdata=[]
Ydata=[]
Xtrain_data=[]
Ytrain_data=[]
Xtest_data=[]
Ytest_data=[]



for idx in range(len(Data)//39):
    if(np.isinf(Data[39*idx:39*(idx+1)][:,:].tolist()).any()):
        print(np.isinf(Data[39*idx:39*(idx+1)][:,:].tolist()).any())
        continue
    if idx< (len(Data)//39)*0.9:
        Xtrain_data.append(Data[39*idx:39*(idx+1)][:,:-1].tolist())
        Ytrain_data.append(Data[39*idx:39*(idx+1)][:,-1].tolist())
    else:
        Xtest_data.append(Data[39*idx:39*(idx+1)][:,:-1].tolist())
        Ytest_data.append(Data[39*idx:39*(idx+1)][:,-1].tolist())
Xtrain_data=np.vstack(Xtrain_data)
Ytrain_data=np.vstack(Ytrain_data)
Xtest_data=np.vstack(Xtest_data)
Ytest_data=np.vstack(Ytest_data)

Ytrain_data=Ytrain_data[:,1:]>Ytrain_data[:,:-1]
Ytrain_data=torch.LongTensor(Ytrain_data)
Ytrain_data=torch.cat((Ytrain_data,torch.zeros(Ytrain_data.shape[0],1)+2),axis=1)
Ytrain_data=Ytrain_data.long()
Ytrain_data=Ytrain_data.view(-1)

Xtrain_data=torch.FloatTensor(Xtrain_data)
Xtest_data=torch.FloatTensor(Xtest_data)

Ytest_data=Ytest_data[:,1:]>Ytest_data[:,:-1]
Ytest_data=torch.LongTensor(Ytest_data)
Ytest_data=torch.cat((Ytest_data,torch.zeros(Ytest_data.shape[0],1)+2),axis=1)

Xtest_data=np.vstack(Xtest_data)
Ytest_data=np.vstack(Ytest_data)
Xtest_data=torch.FloatTensor(Xtest_data)
Ytest_data=torch.LongTensor(Ytest_data)
Ytest_data=Ytest_data.view(-1)

Ytrain_data=Ytrain_data.T
Ytest_data=Ytest_data.T


# In[3]:


bptt = 39
TGT_VOCAB_SIZE = 3
EMB_SIZE =128
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


# In[4]:


torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_VOCAB_SIZE = Xtrain_data.shape[1]


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
summary = SummaryWriter()
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
Val_loss=[]
Train_loss=[]
Accuracy=[]
F1score=[]
NUM_EPOCHS = 3000
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


PATH='best_model_Trans_seq_'+date_time+'_midprice'

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


import pickle
with open("Val_loss_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Val_loss, fp)
    
with open("Train_loss_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Train_loss, fp)
    
with open("Accuracy_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(Accuracy, fp)
    
with open("F1_Trans_Mid"+date_time, "wb") as fp:   #Pickling
    pickle.dump(F1score, fp)


# import matplotlib.pyplot as plt
# plt.plot(Val_loss);
# plt.plot(Train_loss);
# 

# plt.plot(F1score)
# plt.plot(Accuracy)
# # plt.legend(['F1sc','Acc'])

# 
