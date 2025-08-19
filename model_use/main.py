from model_use.simpleNN import create_model as simpleNN

def choose_model(name , test_person , emotion) : 
    if name == 'simpleNN' : 
        return simpleNN(test_person , emotion)

