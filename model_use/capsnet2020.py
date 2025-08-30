from dataset.main import data
import torch 
import torch.nn as nn 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.capsnet2020 import model
from train import Trainer
import torch

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
    time_len = 2 
    num_filter = 128
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
    Model = model (num_filter, num_channel, time_len, caps_len, num_emotions, out_dim)
    unique_Loss_fn = lambda v , y : loss_fn(v , y)
    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=2e-4,
        epochs=30,
        loss_fn = unique_Loss_fn, 
        checkpoint_path=f"eeg_checkpoint{fold_idx}.pth",
        log_path=f"eeg_log{fold_idx}.json", 
    )
    #____fit_model_____#
    return  trainer.fit()
















