from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
test_person = [2 , 15]
overlap = 0.25
time_len = 10
# تعیین device به صورت صحیح
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion= 'arousal'
label_method = 'binary'
batch_size = 126 

# نمونه‌سازی کلاس
my_dataset = data(test_person, overlap, time_len, device, emotion, label_method, batch_size)

# حلقه زدن روی DataLoader که از متد train_data() برمی‌گردد.
for x_train, y_train in my_dataset.train_data():
    print(x_train.shape)
    print(y_train.shape)