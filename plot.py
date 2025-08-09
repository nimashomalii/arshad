import matplotlib.pyplot as plt
import torch

def plot_training_history(history):
    # بررسی و تبدیل تنسور به numpy
    for key in history:
        if isinstance(history[key], torch.Tensor):
            history[key] = history[key].detach().cpu().numpy()
    
    epochs = range(1, len(history['train_loss']) + 1)

    # --- نمودار Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")

    # --- نمودار Accuracy ---
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
