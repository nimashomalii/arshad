from kfold_validation import validate
import sys
from plot import plot_training_history
k = sys.argv[2]
model_name  = sys.argv[1]
train_loss , val_loss , train_acc , val_acc = validate(model_name , k , 23)
history = {
    'train_loss' : train_loss , 
    'val_loss' : val_loss , 
    'train_acc' : train_acc , 
    'val_acc' : val_acc
}
plot_training_history(history)
