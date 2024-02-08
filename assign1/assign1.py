"""
Author: William Noonan
Neural Networks
Assignment 1
"""

import numpy as np
import matplotlib.pyplot as plt

# load the data
learn_rate = np.load('data/learning_rate.npy')
train_hist = np.load('data/train_hist.npy')
val_hist = np.load('data/val_hist.npy')


# learning rate, get points marking end of epoch
epoch_i = [i * 391 for i in range(1, 10)]
index_val = [(i, x) for i, x in enumerate(learn_rate)]
epoch_pts = [index_val[i-1] for i in epoch_i]

# loss, accuracy, f1 data
loss_train = train_hist[:, 0]
loss_val = val_hist[:, 0]

acc_train = train_hist[:, 1]
acc_val = val_hist[:, 1]

f1_train = train_hist[:, 2]
f1_val = val_hist[:, 2]

# Plots
fig,a = plt.subplots(2,2, figsize=(10, 10))

# Learning Rate plot
a1 = a[0][0]
a1.plot(learn_rate)
a1.scatter(*zip(*epoch_pts), marker='.')
a1.legend(['learning rate','epoch marker'], loc='upper right')
a1.xaxis.set_ticks(np.arange(0, 4000, 500))
a1.set_xticklabels(a1.get_xticklabels(), rotation=30)
a1.set(xlabel='Training Steps', ylabel='Learning Rate')

# Loss plot
a2 = a[0][1]
a2.plot(loss_train, ls='-', marker='.')
a2.plot(loss_val, ls='-', marker='.')
a2.legend(['train', 'val'], loc='upper right')
a2.set_xticks(range(len(loss_train)))  # must set number of xticks before using custom strings
a2.set_xticklabels([str(x+1) for x in range(len(loss_train))], rotation=30)
a2.set(xlabel='Epochs', ylabel='Loss')

# Accuracy plot
a3 = a[1][0]
a3.plot(acc_train, ls='-', marker='.')
a3.plot(acc_val, ls='-', marker='.')
a3.legend(['train', 'val'], loc='upper right')
a3.set_xticks(range(len(acc_train)))  # must set number of xticks before using custom strings
a3.set_xticklabels([str(x+1) for x in range(len(acc_train))], rotation=30)
a3.set(xlabel='Epochs', ylabel='Accuracy')

# F1 Score plot
a4 = a[1][1]
a4.plot(f1_train, ls='-', marker='.')
a4.plot(f1_val, ls='-', marker='.')
a4.legend(['train', 'val'], loc='upper right')
a4.set_xticks(range(len(f1_train)))  # must set number of xticks before using custom strings
a4.set_xticklabels([str(x+1) for x in range(len(f1_train))], rotation=30)
a4.set(xlabel='Epochs', ylabel='F1 Score')

# these must be last
fig.tight_layout()
plt.savefig('assign1plot.png')
plt.show()

