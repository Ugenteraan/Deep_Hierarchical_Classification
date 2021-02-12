'''
Graph plotting functions.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fig = plt.figure(figsize=(20, 5))

def plot_loss_acc(path, num_epoch, train_accuracies_superclass, train_accuracies_subclass, train_losses,
                    test_accuracies_superclass, test_accuracies_subclass, test_losses):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''

    plt.clf()

    epochs = [x for x in range(num_epoch+1)]

    train_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_superclass, "Mode":['train']*(num_epoch+1)})
    train_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_subclass, "Mode":['train']*(num_epoch+1)})
    test_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_superclass, "Mode":['test']*(num_epoch+1)})
    test_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_subclass, "Mode":['test']*(num_epoch+1)})

    data_superclass = pd.concat([train_superclass_accuracy_df, test_superclass_accuracy_df])
    data_subclass = pd.concat([train_subclass_accuracy_df, test_subclass_accuracy_df])

    sns.lineplot(data=data_superclass, x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Superclass Accuracy Graph')
    plt.savefig(path+f'accuracy_superclass_epoch.png')

    plt.clf()

    sns.lineplot(data=data_subclass, x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Subclass Accuracy Graph')
    plt.savefig(path+f'accuracy_subclass_epoch.png')

    plt.clf()

    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses, "Mode":['train']*(num_epoch+1)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses, "Mode":['test']*(num_epoch+1)})

    data = pd.concat([train_loss_df, test_loss_df])

    sns.lineplot(data=data, x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')

    plt.savefig(path+f'loss_epoch.png')

    return None
