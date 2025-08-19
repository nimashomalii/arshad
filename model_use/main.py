from model_use.simpleNN import create_model as simpleNN

def choose_model(name ,emotion ,  test_person , fold_idx ) : 
    if name == 'simpleNN' : 
        return simpleNN(test_person , emotion , fold_idx)



