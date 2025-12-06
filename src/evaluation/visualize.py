import matplotlib.pyplot as plt
import os

def plot_history(history, out_path='outputs/plots'):
    os.makedirs(out_path, exist_ok=True)
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.history.get('val_acc') if hasattr(history, 'history') else None

    plt.figure()
    plt.plot(loss, label='loss')
    if val_loss: plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss_curve.png'))
    plt.close()

    if acc:
        plt.figure()
        plt.plot(acc, label='acc')
        if val_acc: plt.plot(val_acc, label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(out_path, 'acc_curve.png'))
        plt.close()
