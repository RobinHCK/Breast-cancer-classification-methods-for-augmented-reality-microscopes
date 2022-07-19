from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import seaborn as sn
import pandas as pd
import numpy as np



# Get accuracy for each network
content_csv_acc = ""
content_matrix_acc = [[]]

for mg_train in mgs + ['all']:
    for mg_test in mgs:
        accuracy = 0
        for fold in folds:
            with open(("results/classif_report_" + str(mg_train) + "X_" + str(mg_test) + "X_K" + str(fold) + "_" + str(width) + "x" + str(height) + ".pkl"), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = "latin1"
                classif_report = u.load()
                accuracy += classif_report["accuracy"]
                f.close()

        content_csv_acc += str(accuracy/5) + "," # 5-K cross validation
        content_matrix_acc[-1].append(accuracy/5)
    content_csv_acc = content_csv_acc[:-1] + "\n"
    content_matrix_acc.append([])
content_matrix_acc = np.array(content_matrix_acc[:-1])

# Get standard deviation for each network
content_csv_std = ""
content_matrix_std = [[]]

for mg_train in mgs + ['all']:
    for mg_test in mgs:
        accuracy = []
        for fold in folds:
            with open(("results/classif_report_" + str(mg_train) + "X_" + str(mg_test) + "X_K" + str(fold) + "_" + str(width) + "x" + str(height) + ".pkl"), 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = "latin1"
                classif_report = u.load()
                accuracy.append(classif_report["accuracy"])
                f.close()

        content_csv_std += str(np.std(accuracy)) + "," 
        content_matrix_std[-1].append(np.std(accuracy))
    content_csv_std = content_csv_std[:-1] + "\n"
    content_matrix_std.append([])
content_matrix_std = np.array(content_matrix_std[:-1])



# Save data to draw accuracy heatmap as csv file
f = open("results/heatmap_acc.csv", "w")
f.write(content_csv_acc)
f.close()

print('Accuracy heatmap values saved')

# Save data to draw std heatmap as csv file
f = open("results/heatmap_std.csv", "w")
f.write(content_csv_std)
f.close()

print('Std heatmap values saved')



# Create & Save heatmap of accuracies and standard deviation
mgs_test = [str(mg) + "X" for mg in mgs]
mgs_train = mgs_test + ["all X"]

cmap = LinearSegmentedColormap.from_list(name='test', colors=['white','#17588e'])

plt.figure(figsize=(20,20))
content_matrix_acc_std = (np.asarray(["{0:.1}\n±{0:.1}".format(content_matrix_acc, content_matrix_std) for content_matrix_acc, content_matrix_std in zip(content_matrix_acc.flatten(), content_matrix_std.flatten())])).reshape(len(mgs_train), len(mgs_test))
sn.set(font_scale=2)
sn.heatmap(content_matrix_acc, annot=content_matrix_acc_std, annot_kws={'size': 10}, fmt="", cmap=cmap, vmin=np.amin(content_matrix_acc), vmax=np.amax(content_matrix_acc), xticklabels=mgs_test, yticklabels=mgs_train)
plt.title("Accuracy heatmap: network trained and tested on different magnification level", fontsize=20)
plt.xlabel('Used to test', fontsize=20)
plt.ylabel('Used to train', fontsize=20)
plt.savefig("results/heatmap_acc.pdf")
plt.clf()
plt.cla()

print('Accuracy heatmap saved')
