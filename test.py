from config import *
import sys
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time



# Arguments
# python test.py mg_train mg_test num_fold width height
mg_train = sys.argv[1]
mg_test = sys.argv[2]
num_fold = sys.argv[3]

print("Run " + sys.argv[0] + " train on " + mg_train + "X test on " + mg_test + "X K" + num_fold + " " + str(width) + "x" + str(height) + "px")

# Load data
print("Loading data")
x_test, y_test = [], []

Folds_csv = open("../archive/Folds.csv", "r").read()
Folds_lines = Folds_csv.split('\n')

# Build a dataset of a given magnification level from 40X, 100X, 200X and 400X magnification levels
for line in Folds_lines:

    nearest_mg = 40
    if int(mg_test) > 40:
        nearest_mg = 100
    elif int(mg_test) > 100:
        nearest_mg = 200
    elif int(mg_test) > 200:
        nearest_mg = 400

    if len(line) > 0 and line.split(',')[1] == str(nearest_mg) and line[0] == num_fold:

        img_path = line.split(',')[3]
        img_grp = line.split(',')[2]
        if line.split(',')[3].split('/')[3] == 'benign':
            img_class = 0
        elif line.split(',')[3].split('/')[3] == 'malignant':
            img_class = 1

        crop_size_width = int(width*(nearest_mg/int(mg_test)))
        crop_size_height = int(height*(nearest_mg/int(mg_test)))

        x1 = int((700-crop_size_width)/2)
        y1 = int((460-crop_size_height)/2)
        x2 = int((700-crop_size_width)/2) + crop_size_width
        y2 = int((460-crop_size_height)/2) + crop_size_height

        img_crop = load_img("../archive/" + img_path).crop((x1,y1,x2,y2))
        img_resized = img_crop.resize((width, height))
        img = img_to_array(img_resized)

        if img_grp == 'test':
            x_test.append(img)
            y_test.append(img_class)
            
x_test = np.array(x_test)

print("X - test: ", x_test.shape)

# Categorize the labels
y_test = to_categorical(y_test, num_classes)

print("Y - test: ", y_test.shape)

# Model to test
model = load_model("results/model_" + mg_train + "X_K" + num_fold + "_" + str(width) + "x" + str(height) + ".h5")

# Predict & Compute inference time
start_time_predict = time.time()
y_pred = model.predict(x_test*rescale, batch_size=batch_size, verbose=1)
end_time_predict = time.time()
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Inference time (FPS_720p ; FPS_1080p)
nbr_pixels_in_patch = width*height
nbr_patches_predicted = len(y_pred)
nbr_seconds_to_predict = end_time_predict-start_time_predict
nbr_pixels_720p = 720*1280
FPS_720p = (nbr_patches_predicted / nbr_seconds_to_predict) / (nbr_pixels_720p / nbr_pixels_in_patch)
nbr_pixels_1080p = 1080*1920
FPS_1080p = (nbr_patches_predicted / nbr_seconds_to_predict) / (nbr_pixels_1080p / nbr_pixels_in_patch)
content_inference_file = str(FPS_720p) + ";" + str(FPS_1080p)

open("results/inference_" + mg_train + "X_" + mg_test + "X_K" + num_fold + ".csv", "w").write(content_inference_file)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(conf_matrix)

# Save confusion Matrix to pickle file
pickle.dump(conf_matrix, open("results/conf_matrix_" + mg_train + "X_" + mg_test + "X_K" + num_fold + "_" + str(width) + "x" + str(height) + ".pkl", 'wb'))

print('Confusion Matrix saved')

# Classification Report
classif_report = classification_report(y_test, y_pred, target_names=['benign','malignant'], output_dict=True)
print('Classification Report')
print(classif_report)

# Save classification Report to pickle file
pickle.dump(classif_report, open("results/classif_report_" + mg_train + "X_" + mg_test + "X_K" + num_fold + "_" + str(width) + "x" + str(height) + ".pkl", 'wb'))

print('Classification Report saved')
