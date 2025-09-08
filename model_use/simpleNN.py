from dataset.main import data , data_for_subject_dependet
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.simpleNN import model
from train import Trainer
import random
from functions import k_fold_data_segmentation
from  torch.utils.data import DataLoader , TensorDataset
import numpy as np 

#____Model______#
def create_model(test_person , emotion,category , fold_idx ) : 
    overlap = 0.1
    time_len = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 126 
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, category, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()
    Model = model([8960, 64, output_dim])  # معماری دلخواه

    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=1e-3,
        epochs=50,
        checkpoint_path=f"eeg_checkpoint{fold_idx}.pth",
        log_path=f"eeg_log{fold_idx}.json", 
    )
    #____fit_model_____#
    return  trainer.fit()

def subject_dependent_validation (emotion ,category, fold_idx , k=5) : 
    overlap = 0.05
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 128
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
            Model = model([1792, 64, output_dim])  # معماری دلخواه        
            #____trainer_______#
            trainer = Trainer(
                model=Model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                label_method=category,
                optimizer_cls=torch.optim.Adam,
                lr=1e-3,
                epochs=30,
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
