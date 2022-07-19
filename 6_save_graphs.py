import os
import pickle
import matplotlib.pyplot as plt
from config import *

# Create & Save graphs for each network :
#  - Training and Validation loss
#  - Training and Validation accuracy
for mg_train in mgs:
    for fold in folds:
        with open(("results/history_" + str(mg_train) + "X_K" + str(fold) + "_" + str(width) + "x" + str(height) + ".pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            history = u.load()

            loss_train = history['loss']
            loss_val = history['val_loss']
            epochs = range(1,len(loss_train)+1)
            plt.plot(epochs, loss_train, 'g', label='Training loss')
            plt.plot(epochs, loss_val, 'b', label='validation loss')
            plt.title('Training and Validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('results/loss_' + str(mg_train) + 'X_K' + str(fold) + '.pdf')
            plt.clf()

            acc_train = history['accuracy']
            acc_val = history['val_accuracy']
            epochs = range(1,len(acc_train)+1)
            plt.plot(epochs, acc_train, 'g', label='Training accuracy')
            plt.plot(epochs, acc_val, 'b', label='validation accuracy')
            plt.title('Training and Validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend()
            plt.savefig('results/acc_' + str(mg_train) + 'X_K' + str(fold) + '.pdf')
            plt.clf()

            f.close()
