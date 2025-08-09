import matplotlib.pyplot as plt

def plot_training_history(history):
    history = {
                "epoch": [1 , 5 , 1 , 2 , 4 ],
                "train_loss":[0.1 , 0.2 , 0.3 , 0.8 , 0.1],
                "val_loss": [0.8 , 0.7 , 0.6 , 0.3 , 0.7],
                "train_acc": [0.1 , 0.2 , 0.3 , 0.8 , 0.1] , 
                "val_acc": [0.8 , 0.7 , 0.6 , 0.3 , 0.7]
            }

    epochs = range(1, len(history['train_loss']) + 1)

    # نمودار Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # نمودار Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# استفاده بعد از فیت کردن
#plot_training_history(history)

