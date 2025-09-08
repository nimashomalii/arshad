from model_use.simpleNN import create_model as simpleNN 
from model_use.simpleNN import subject_dependent_validation as simpleNN_sub_dep
from model_use.cnn_45138  import create_model as cnn_45138
from model_use.cnn_45138 import subject_dependent_validation as cnn_45138_sub_dep
from model_use.capsnet2020 import create_model as capsnet2020
from model_use.capsnet2020 import subject_dependent_validation as capsnet2020_sub_dep

def choose_model(name ,emotion , category,  test_person , fold_idx  , subject_dependecy = 'subject_independent') : 
   
    if (name == 'simpleNN') & (subject_dependecy == 'subject_independent') : 
        return simpleNN(test_person , emotion ,category, fold_idx)
    if (name == 'simpleNN') & (subject_dependecy == 'subject_dependent') : 
        return simpleNN_sub_dep(emotion ,category, fold_idx)


    if (name == 'cnn_45138') & (subject_dependecy == 'subject_independent') : 
        return cnn_45138(test_person , emotion ,category, fold_idx)
    if (name == 'cnn_45138') & (subject_dependecy == 'subject_dependent') : 
        return cnn_45138( emotion ,category, fold_idx)


    if (name == 'capsnet2020') & (subject_dependecy == 'subject_independent') : 
        return capsnet2020(test_person , emotion ,category, fold_idx)
    if (name == 'capsnet2020') & (subject_dependecy == 'subject_dependent') : 
        return capsnet2020( emotion ,category, fold_idx)






