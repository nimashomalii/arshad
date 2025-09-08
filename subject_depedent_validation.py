from model_use.main import choose_model
import sys
from plot import plot_subject_dependet
import numpy as np
emotion = sys.argv[2]
category  = sys.argv[3]
k = int(sys.argv[4])
model_name  = sys.argv[1]
accuraceis = choose_model(model_name ,emotion , category, None , None  , subject_dependecy = 'subject_dependent')
print(f'''
    average test accuracy :  {np.sum(accuraceis['test'])/23}
    average train accuracy : {np.sum(accuraceis['train'])/23}
''') 
plot_subject_dependet(accuraceis)
