import random
import sys
from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from train import Trainer
import torch
import plot 
from model_use.main import choose_model
import numpy as np 

def k_fold_validation(k, num_people=23):
    patients = list(range(num_people))
    random.shuffle(patients)
    
    fold_size = num_people // k
    remainder = num_people % k
    
    kfold_patients = []
    start = 0
    
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0) 
        kfold_patients.append(patients[start:end])
        start = end
    
    return kfold_patients

def validate(model_name, emotion , k , num_people= 23) : 
    pateints = k_fold_validation(k , num_people)
    len_patients = len(pateints)
    i = 0 
    for test_person in pateints :
        print("a new procedure is takeing place . . . " ) 
        history = choose_model(model_name, emotion , test_person)
        if i ==0 : 
            train_loss = np.array(history['train_loss'])
            val_loss = np.array(history['val_loss'])
            train_acc = np.array(history['train_acc'])
            val_acc  = np.array(history['val_acc'])
        else : 
            train_loss += np.array(history['train_loss'])
            val_loss += np.array(history['val_loss'])
            train_acc += np.array(history['train_acc'])
            val_acc  += np.array(history['val_acc'])
        i+=1
        print(history['train_loss'])
    train_loss /= len_patients
    val_loss /= len_patients
    train_acc /= len_patients
    val_acc /= len_patients
    return train_loss , val_loss , train_acc , val_acc

    



