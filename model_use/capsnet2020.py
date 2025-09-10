from dataset.main import data
import torch 
import torch.nn as nn 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.capsnet2020 import model
from train import Trainer
import torch
from dataset.main import data , data_for_subject_dependet
from train import Trainer
import random
from functions import k_fold_data_segmentation
from  torch.utils.data import DataLoader , TensorDataset
import numpy as np 


def loss_fn (v , y , landa=0.5 , m_plus=0.9 , m_mines=0.1) :  #v:  (B, M) y:(B)
    relu = nn.ReLU()
    total_loss  = 0 
    for i in range(v.shape[1]) : 
        T = (y == i ).float()
        Loss =  T * (relu(m_plus - v[: , i]))**2 + landa*(1-T)*(relu(v[: , i] -  m_mines))**2
        total_loss += Loss
    return total_loss.sum()


#____Model______#                          categy ; binary or 5category
def create_model(test_person , emotion,category , fold_idx ) : 
    overlap = 0
    time_len = 1 
    num_filter =256
    num_channel = 14 
    caps_len = 8
    out_dim= 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    num_emotions = output_dim
    batch_size =256
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, category, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()
    Model = model (num_filter, 128* time_len, caps_len, num_emotions, out_dim)
    unique_Loss_fn = lambda v , y : loss_fn(v , y)
    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=2e-5,
        epochs=30,
        loss_fn = unique_Loss_fn, 
        checkpoint_path=f"eeg_checkpoint{fold_idx}.pth",
        log_path=f"eeg_log{fold_idx}.json", 
    )
    #____fit_model_____#
    return  trainer.fit()


def subject_dependent_validation (emotion ,category, fold_idx , k=5) : 
    num_filter =12
    num_channel = 14 
    caps_len = 8
    out_dim= 16
    overlap = 0
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    num_emotions = output_dim
    batch_size = 100
    data_type = torch.float32
    accuracies_on_subjects  = {
        'train' : [] , 
        'test' : []
    } 
    for person_num in range(23) : 
        fold_idx = 0
        for (x_train , x_test , y_train , y_test) in data_for_subject_dependet(overlap , time_len , emotion , category , data_type , device ,person_num , k): 
            print(f'''
                        the size of the x_train is : {x_train.shape[0]}
            ''')
            test_dataset = TensorDataset(x_test , y_test)
            test_loader = DataLoader(test_dataset ,batch_size , shuffle=True )
            train_dataset = TensorDataset(x_train , y_train )
            train_loader = DataLoader(train_dataset , batch_size,shuffle=True )
            Model = model (num_filter, 128* time_len, caps_len, num_emotions, out_dim)
            unique_Loss_fn = lambda v , y : loss_fn(v , y) # معماری دلخواه        
            #____trainer_______#
            trainer = Trainer(
                model=Model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                label_method=category,
                optimizer_cls=torch.optim.Adam,
                lr=5e-5,
                epochs=30,
                loss_fn = unique_Loss_fn, 
                checkpoint_path=f"eeg_checkpoint{fold_idx + person_num*5}.pth",
                log_path=f"eeg_log{fold_idx + person_num*5}.json", 
            )
            #____fit_model_____#
            history =  trainer.fit()
            if fold_idx ==0 : 
                train_loss = np.array(history['train_loss'])
                val_loss = np.array(history['val_loss'])
                train_acc = np.array(history['train_acc'])
                val_acc = np.array(history['val_acc'])
            else : 
                train_loss += np.array(history['train_loss'])
                val_loss += np.array(history['val_loss'])
                train_acc += np.array(history['train_acc'])
                val_acc += np.array(history['val_acc'])
            fold_idx +=1
        person_num +=1
        train_acc  /=k
        train_loss /=k
        val_loss   /=k
        val_acc    /=k

        accuracies_on_subjects['train'].append(np.max(np.array(train_acc)))
        accuracies_on_subjects['test'].append(np.max(np.array(val_acc)))
    return accuracies_on_subjects






