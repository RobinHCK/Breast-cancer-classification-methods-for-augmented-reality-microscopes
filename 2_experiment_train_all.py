import os
from config import *



# Train one network on all magnification levels and per fold
for fold in folds:
    command = "python train_all.py " + str(fold)
    os.system(command)
        
    print("Run command " + command)
