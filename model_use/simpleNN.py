from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.simpleNN import model
from train import Trainer
import torch

#____Model______#
def create_model(test_person , emotion , fold_idx) : 
    overlap = 0.1
    time_len = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_method = 'binary'
    batch_size = 126 
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, label_method, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()
    Model = model([8960, 64, 1])  # معماری دلخواه

    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer_cls=torch.optim.Adam,
        lr=1e-3,
        epochs=50,
        checkpoint_path="eeg_checkpoint.pth",
        log_path="eeg_log.json"
    )
    #____fit_model_____#
    return  trainer.fit()


