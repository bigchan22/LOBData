import numpy as np
import torch
def Data_train_load(Data_train):
    Xtrain_data=[]
    Ytrain_data=[]
    for idx in range(len(Data_train)//39):
        if(np.isinf(Data_train[39*idx:39*(idx+1)][:,:].tolist()).any()):
            print(np.isinf(Data_train[39*idx:39*(idx+1)][:,:].tolist()).any())
            continue
        Xtrain_data.append(Data_train[39*idx:39*(idx+1)][:,:].tolist())
        Ytrain_data.append(Data_train[39*idx:39*(idx+1)][:,-1].tolist())

    Xtrain_data=np.vstack(Xtrain_data)
    Ytrain_data=np.vstack(Ytrain_data)
    Xtrain_data=torch.FloatTensor(Xtrain_data)
    Ytrain_data=torch.IntTensor(Ytrain_data)
    Ytrain_data=Ytrain_data.view(-1)
    Ytrain_data=torch.cat(((Ytrain_data[1:]-Ytrain_data[:-1]),torch.tensor(0).unsqueeze(-1)))
    Ytrain_data[::39]=0



    Ytrain_data=2*(Ytrain_data>0).long()+(Ytrain_data==0).long()
    Ytrain_data=Ytrain_data.T

    return Xtrain_data, Ytrain_data


def Data_test_load(Data_test):
    Xtest_data=[]
    Ytest_data=[]

    for idx in range(len(Data_test)//39):
        if(np.isinf(Data_test[39*idx:39*(idx+1)][:,:].tolist()).any()):
            print(np.isinf(Data_test[39*idx:39*(idx+1)][:,:].tolist()).any())
            continue
        Xtest_data.append(Data_test[39*idx:39*(idx+1)][:,:].tolist())
        Ytest_data.append(Data_test[39*idx:39*(idx+1)][:,-1].tolist())

    Xtest_data=np.vstack(Xtest_data)
    Ytest_data=np.vstack(Ytest_data)
    Xtest_data=torch.FloatTensor(Xtest_data)
    Ytest_data=torch.IntTensor(Ytest_data)
    Ytest_data=Ytest_data.view(-1)
    Ytest_data=torch.cat(((Ytest_data[1:]-Ytest_data[:-1]),torch.tensor(0).unsqueeze(-1)))
    Ytest_data[::39]=0

    Ytest_data=2*(Ytest_data>0).long()+(Ytest_data==0).long()
    Ytest_data=Ytest_data.T
    return Xtest_data,Ytest_data

def Data_load(Data_train,Data_test):
    Xtrain_data,Ytrain_data=Data_train_load(Data_train)
    Xtest_data,Ytest_data=Data_test_load(Data_test)    
    return Xtrain_data,Ytrain_data,Xtest_data,Ytest_data
#     Xdata=[]
#     Ydata=[]
#     Xtrain_data=[]
#     Ytrain_data=[]
#     Xtest_data=[]
#     Ytest_data=[]



#     for idx in range(len(Data_train)//39):
#         if(np.isinf(Data_train[39*idx:39*(idx+1)][:,:].tolist()).any()):
#             print(np.isinf(Data_train[39*idx:39*(idx+1)][:,:].tolist()).any())
#             continue
#         Xtrain_data.append(Data_train[39*idx:39*(idx+1)][:,:].tolist())
#         Ytrain_data.append(Data_train[39*idx:39*(idx+1)][:,-1].tolist())
#     for idx in range(len(Data_test)//39):
#         if(np.isinf(Data_test[39*idx:39*(idx+1)][:,:].tolist()).any()):
#             print(np.isinf(Data_test[39*idx:39*(idx+1)][:,:].tolist()).any())
#             continue
#         Xtest_data.append(Data_test[39*idx:39*(idx+1)][:,:].tolist())
#         Ytest_data.append(Data_test[39*idx:39*(idx+1)][:,-1].tolist())

#     Xtrain_data=np.vstack(Xtrain_data)
#     Ytrain_data=np.vstack(Ytrain_data)
#     Xtrain_data=torch.FloatTensor(Xtrain_data)
#     Ytrain_data=torch.IntTensor(Ytrain_data)
#     Ytrain_data=Ytrain_data.view(-1)
#     Ytrain_data=torch.cat(((Ytrain_data[1:]-Ytrain_data[:-1]),torch.tensor(0).unsqueeze(-1)))
#     Ytrain_data[::39]=0

#     Xtest_data=np.vstack(Xtest_data)
#     Ytest_data=np.vstack(Ytest_data)
#     Xtest_data=torch.FloatTensor(Xtest_data)
#     Ytest_data=torch.IntTensor(Ytest_data)
#     Ytest_data=Ytest_data.view(-1)
#     Ytest_data=torch.cat(((Ytest_data[1:]-Ytest_data[:-1]),torch.tensor(0).unsqueeze(-1)))
#     Ytest_data[::39]=0


#     Ytrain_data=2*(Ytrain_data>0).long()+(Ytrain_data==0).long()
#     Ytest_data=2*(Ytest_data>0).long()+(Ytest_data==0).long()
#     Ytrain_data=Ytrain_data.T
#     Ytest_data=Ytest_data.T


