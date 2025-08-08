import torch
import json
import sys
from dataset.extractor import DataExtractor 
from dataset.make_variable import dataset

from  torch.utils.data import DataLoader , TensorDataset

import time

import torch.nn as nn 
# در فایل main.py
def prepar_dataset(test_person, over_lap, time_len, device, emotion, label_method=None):
    with open('dataset/config.json', 'r') as f:
        config = json.load(f)
    file_id = config['file_id']
    file_path = config['data_paths']

    extract_data = DataExtractor()
    extract_data.extract_data_file(file_id)

    data_manage = dataset(test_person, over_lap, time_len, device, emotion, label_method)
    data_manage.extract(file_path, torch.float64)
    data_manage.normalize()
    x_train, x_test, y_train, y_test = data_manage.receive_data()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    extract_data.clean_extracted_data()

    return x_train, x_test, y_train, y_test
class data : 
    def __init__(self , test_person, overlap, time_len, device, emotion, label_method, batch_size ) : 
        self.x_train, self.x_test, self.y_train,  self.y_test = prepar_dataset(test_person, overlap, time_len, device, emotion, label_method)
        test_dataset = TensorDataset(self.x_test , self.y_test)
        self.test_loader = DataLoader(test_dataset ,batch_size , shuffle=True )
        train_dataset = TensorDataset(self.x_train , self.y_train )
        self.train_loader = DataLoader(train_dataset , batch_size,shuffle=False )
    def train_data(self ) :
        return self.train_loader
    def  test_data(self ) : 
        return self.test_loader









