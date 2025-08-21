from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.cnn_45138 import model
from train import Trainer
import torch


#____Model______#                          categy ; binary or 5category
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
    Model = model(time_len=time_len  , num_output= output_dim)

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



