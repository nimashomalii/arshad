from model_use.simpleNN import create_model as simpleNN
from model_use.cnn_45138  import create_model as cnn_45138

def choose_model(name ,emotion , category,  test_person , fold_idx ) : 
    if name == 'simpleNN' : 
        return simpleNN(test_person , emotion ,category, fold_idx)
    if name == 'cnn_45138' : 
        return cnn_45138(test_person , emotion ,category, fold_idx)



