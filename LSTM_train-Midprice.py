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
from source.Attention_LSTM import RNNModel
from source.train import train_epoch_lstm,evaluate_lstm


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


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
        Xtrain_data.append(Data[39*idx:39*(idx+1)][:,:].tolist())
        Ytrain_data.append(Data[39*idx:39*(idx+1)][:,-1].tolist())
    else:
        Xtest_data.append(Data[39*idx:39*(idx+1)][:,:].tolist())
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


# In[4]:


bptt = 39
TGT_VOCAB_SIZE = 3
EMB_SIZE =128
NHEAD = 16
FFN_HID_DIM = 128
BATCH_SIZE = 32
NUM_LAYERS = 3


# In[5]:


Val_loss=[]
Train_loss=[]
Accuracy=[]
F1score=[]


torch.manual_seed(0)

SRC_VOCAB_SIZE = Xtrain_data.shape[1]


lstm = RNNModel(rnn_type='LSTM',ntoken=SRC_VOCAB_SIZE,ninp=EMB_SIZE,nhid=FFN_HID_DIM,nlayers=NUM_LAYERS,proj_size=TGT_VOCAB_SIZE,
                attention=False)
summary = SummaryWriter()
for p in lstm.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

lstm = lstm.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

NUM_EPOCHS = 4000
best_val_loss=100000000
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss,_ = train_epoch_lstm(lstm, optimizer,Xtrain_data,Ytrain_data,loss_fn,device,BATCH_SIZE,bptt)
    end_time = timer()
    val_loss,acc,prec,reca,f1sc,confusion = evaluate_lstm(lstm,Xtest_data,Ytest_data,loss_fn,device,BATCH_SIZE,bptt)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_confusion=confusion
        best_acc=acc
        best_prec=prec
        best_reca=reca
        best_f1sc=f1sc
        best_model = lstm
    Val_loss.append(val_loss)
    Train_loss.append(train_loss)
    Accuracy.append(acc)
    F1score.append(f1sc)
now = datetime.now()
now.strftime("%m/%d/%Y, %H:%M:%S")

date_time = now.strftime("%m_%d_%Y")


# In[ ]:


PATH='best_model_lstm_seq_'+date_time
#     if featnorm==True:
#         torch.save(best_model.state_dict(), PATH+'norm')
#     else:
#         torch.save(best_model.state_dict(), PATH)



file_name='results/result_LSTM_Mid'+date_time+'.txt'
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


# import matplotlib.pyplot as plt
# plt.plot(Val_loss);
# plt.plot(Train_loss);

# plt.plot(F1score)
# plt.plot(Accuracy)

# In[ ]:


import pickle
with open("Val_loss_LSTM_Mid", "wb") as fp:   #Pickling
    pickle.dump(Val_loss, fp)

    
with open("Train_loss_LSTM_Mid", "wb") as fp:   #Pickling
    pickle.dump(Train_loss, fp)

    
with open("Accuracy_LSTM_Mid", "wb") as fp:   #Pickling
    pickle.dump(Accuracy, fp)

    
with open("F1_LSTM_Mid", "wb") as fp:   #Pickling
    pickle.dump(F1score, fp)


# with open("Val_loss_LSTM_Mid", "rb") as fp:   # Unpickling
#     Val_loss_ALSTM = pickle.load(fp)
# with open("Train_loss_LSTM_Mid", "rb") as fp:   # Unpickling
#     Train_loss_ALSTM = pickle.load(fp)
# with open("Accuracy_LSTM_Mid", "rb") as fp:   # Unpickling
#     Accuracy_ALSTM = pickle.load(fp)
# with open("F1_LSTM_Mid", "rb") as fp:   # Unpickling
#     F1_ALSTM = pickle.load(fp)

# 
# with open("Val_loss_Trans_Mid", "rb") as fp:   # Unpickling
#     Val_loss_Trans = pickle.load(fp)
# 
# with open("Train_loss_Trans_Mid", "rb") as fp:   # Unpickling
#     Train_loss_Trans = pickle.load(fp)
#     
# 
# with open("Accuracy_Trans_Mid", "rb") as fp:   # Unpickling
#     Accuracy_Trans = pickle.load(fp)
#     
# 
# with open("F1_Trans_Mid", "rb") as fp:   # Unpickling
#     F1_Trans = pickle.load(fp)

# with open("Val_loss_ALSTM_Mid", "rb") as fp:   # Unpickling
#     Val_loss_ALSTM = pickle.load(fp)
# 
# with open("Train_loss_ALSTM_Mid", "rb") as fp:   # Unpickling
#     Train_loss_ALSTM = pickle.load(fp)
#     
# 
# with open("Accuracy_ALSTM_Mid", "rb") as fp:   # Unpickling
#     Accuracy_ALSTM = pickle.load(fp)
#     
# 
# with open("F1_ALSTM_Mid", "rb") as fp:   # Unpickling
#     F1_ALSTM = pickle.load(fp)

# plt.plot(Xtest_data[:,-1])

# fig,ax=plt.subplots()
# # ax.plot(Val_loss);
# ax.plot(Train_loss);
# 
# # ax.plot(Val_loss_Trans);
# ax.plot(Train_loss_Trans);
# # ax.plot(Val_loss_ALSTM);
# ax.plot(Train_loss_ALSTM);
# ax.legend(["Train_loss_LSTM","Train_loss_Trans","Train_loss_ALSTM"])
# fig.savefig('Loss.eps')

# plt.plot(Accuracy);
# # plt.plot(F1score);
# 
# plt.plot(Accuracy_Trans);
# plt.plot(Accuracy_ALSTM);
# # plt.plot(F1_Trans);
# plt.legend(["Acc_LSTM","Acc_Trans","Acc_ALSTM"])
# plt.savefig('Accuracy.eps')

# # plt.plot(Accuracy);
# plt.plot(F1score);
# 
# # plt.plot(Accuracy_Trans);
# # plt.plot(Accuracy_ALSTM);
# plt.plot(F1_Trans);
# plt.plot(Accuracy_ALSTM);
# plt.legend(["F1_LSTM","F1_Trans","F1_ALSTM"])
# plt.savefig('F1.eps')

# 
# 
# for MBR_NO,BRN_NO in mbrnlist:
#     if featnorm==True:
#         Data_train = load('Data/Data0930_'+str(MBR_NO)+'_'+str(BRN_NO)+'norm_train.npy',allow_pickle=True)
#         Data_test =  load('Data/Data0930_'+str(MBR_NO)+'_'+str(BRN_NO)+'norm_test.npy',allow_pickle=True)
#     else:
#         Data_train = load('Data/Data0930_'+str(MBR_NO)+'_'+str(BRN_NO)+'train.npy',allow_pickle=True)
#         Data_test =  load('Data/Data0930_'+str(MBR_NO)+'_'+str(BRN_NO)+'test.npy',allow_pickle=True)
# 
#     Xtrain_data,Ytrain_data,Xtest_data,Ytest_data = Data_load(Data_train,Data_test)
#     torch.manual_seed(0)
# 
#     SRC_VOCAB_SIZE = Xtrain_data.shape[1]
# 
# 
#     lstm = RNNModel(rnn_type='LSTM',ntoken=SRC_VOCAB_SIZE,ninp=EMB_SIZE,nhid=FFN_HID_DIM,nlayers=NUM_LAYERS,proj_size=TGT_VOCAB_SIZE,
#                     attention_width=38)
#     summary = SummaryWriter()
#     for p in lstm.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
# 
#     lstm = lstm.to(device)
# 
#     loss_fn = torch.nn.CrossEntropyLoss()
# 
#     optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# 
#     NUM_EPOCHS = 1000
#     best_val_loss=100000000
#     for epoch in range(1, NUM_EPOCHS+1):
#         start_time = timer()
#         train_loss,_ = train_epoch_lstm(lstm, optimizer,Xtrain_data,Ytrain_data,loss_fn,device,BATCH_SIZE,bptt)
#         end_time = timer()
#         val_loss,acc,prec,reca,f1sc,confusion = evaluate_lstm(lstm,Xtrain_data,Ytrain_data,loss_fn,device,BATCH_SIZE,bptt)
#         print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_confusion=confusion
#             best_acc=acc
#             best_prec=prec
#             best_reca=reca
#             best_f1sc=f1sc
#             best_model = lstm
#         summary.add_scalar('val loss', val_loss, epoch)
#         summary.add_scalar('total loss', train_loss, epoch)
#         summary.add_scalar('f1 score', f1sc, epoch)
#         summary.add_scalar('Accuracy', acc, epoch)
#     now = datetime.now()
#     now.strftime("%m/%d/%Y, %H:%M:%S")
# 
#     date_time = now.strftime("%m_%d_%Y")
# 
#     PATH='best_model_alstm_seq_'+date_time+'_'+str(MBR_NO)+'_'+str(BRN_NO)
# #     if featnorm==True:
# #         torch.save(best_model.state_dict(), PATH+'norm')
# #     else:
# #         torch.save(best_model.state_dict(), PATH)
#     if featnorm==True:
#         file_name='results/result_ALSTM_'+date_time+'_norm.txt'
#     else:
#         file_name='results/result_ALSTM_'+date_time+'.txt'
#     text_to_append=PATH+'\t'+"Acc:"+str(best_acc)+'\t'+"prec:"+str(best_prec)+'\t'+"recall:"+str(best_reca)+'\t'+"f1sc:"+str(best_f1sc)
#     print(text_to_append)
#     with open(file_name, "a+") as file_object:
#         # Move read cursor to the start of file.
#         file_object.seek(0)
#         # If file is not empty then append '\n'
#         data = file_object.read(100)
#         if len(data) > 0:
#             file_object.write("\n")
#         # Append text at the end of file
#         file_object.write(text_to_append)
