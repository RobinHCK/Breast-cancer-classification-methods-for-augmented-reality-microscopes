import os
from config import *



# Test every network on every magnification level per fold
for mg_train in mgs:
    for mg_test in mgs:
        for fold in folds:
            command = "python test.py " + str(mg_train) + " " + str(mg_test) + " " + str(fold)
            os.system(command)

            print("Run command " + command)
