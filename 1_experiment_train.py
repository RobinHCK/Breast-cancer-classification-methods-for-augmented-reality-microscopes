import os
from config import *



# Train one network per magnification level and per fold
for mg in mgs:
    for fold in folds:
        command = "python train.py " + str(mg) + " " + str(fold)
        os.system(command)
        
        print("Run command " + command)
