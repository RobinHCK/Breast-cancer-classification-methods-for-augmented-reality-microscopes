import os
from config import *



# Test the networks trained on all magnification levels on every magnification level per fold
for mg_test in mgs:
    for fold in folds:
        command = "python test.py all " + str(mg_test) + " " + str(fold)
        os.system(command)

        print("Run command " + command)
