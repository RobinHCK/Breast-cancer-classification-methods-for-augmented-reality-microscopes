from config import *
import random
random.seed(seed)
import os
import sys
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import pickle



# Arguments
# python train.py mg_train num_fold
mg_train = sys.argv[1]
num_fold = sys.argv[2]

print("Run " + sys.argv[0] + " " + mg_train + "X K" + num_fold + " " + str(width) + "x" + str(height) + "px")

# Load data 
print("Loading data")
x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []

Folds_csv = open("../archive/Folds.csv", "r").read()
Folds_lines = Folds_csv.split('\n')

# Build a dataset of a given magnification level from 40X, 100X, 200X and 400X magnification levels
for line in Folds_lines:

    nearest_mg = 40
    if int(mg_train) > 40:
        nearest_mg = 100
    elif int(mg_train) > 100:
        nearest_mg = 200
    elif int(mg_train) > 200:
        nearest_mg = 400

    if len(line) > 0 and line.split(',')[1] == str(nearest_mg) and line[0] == num_fold:

        img_path = line.split(',')[3]
        img_grp = line.split(',')[2]
        if line.split(',')[3].split('/')[3] == 'benign':
            img_class = 0
        elif line.split(',')[3].split('/')[3] == 'malignant':
            img_class = 1

        crop_size_width = int(width*(nearest_mg/int(mg_train)))
        crop_size_height = int(height*(nearest_mg/int(mg_train)))

        x1 = int((700-crop_size_width)/2)
        y1 = int((460-crop_size_height)/2)
        x2 = int((700-crop_size_width)/2) + crop_size_width
        y2 = int((460-crop_size_height)/2) + crop_size_height

        img_crop = load_img("../archive/" + img_path).crop((x1,y1,x2,y2))
        img_resized = img_crop.resize((width, height))
        img = img_to_array(img_resized)

        if img_grp == 'train':
            x_train.append(img)
            y_train.append(img_class)
        elif img_grp == 'test':
            x_test.append(img)
            y_test.append(img_class)

len_train = len(x_train)

x_val = np.array(x_train)[int(len_train*0.85):] # 10%
x_train = np.array(x_train)[:int(len_train*0.85)] # 55%
x_test = np.array(x_test) # 35%

print("X - train: ", x_train.shape, " | val: ", x_val.shape, " | test: ", x_test.shape)

# Categorize the labels
y_val = to_categorical(np.array(y_train)[int(len_train*0.85):], num_classes)
y_train = np.array(y_train)[:int(len_train*0.85)]
y_test = to_categorical(y_test, num_classes)

print("Y - train: ", y_train.shape, " | val: ", y_val.shape, " | test: ", y_test.shape)

# Create the base pre-trained model
base_model = InceptionResNetV2(input_shape=(height, width, 3), include_top=False, weights=weights)

x = base_model.output
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(dropout_rate, seed=seed)(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
#print(model.summary())

k = 5 # number of end layers to retrain 
layers = base_model.layers[:-k] if k != 0 else base_model.layers
for layer in layers: 
    layer.trainable = False

# Compile model
opt = SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(loss=loss, optimizer=opt, metrics=metrics)

x_train, y_train = shuffle(x_train, y_train, random_state=seed)

y_train = to_categorical(y_train, num_classes)

# Initiate the train and validation generators with data Augumentation
train_datagen = ImageDataGenerator(rescale=rescale, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
train_datagen.fit(x_train)
generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=rescale)
val_datagen.fit(x_val)
val_generator = val_datagen.flow(x_val,  y_val, batch_size=batch_size)

# Train the model
#class_weights = {0: (n_samples/(2*n_samples_benign)), 1: (n_samples/(2*n_samples_malign))}
earlystopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=earlystopping_patience, verbose=1, mode='max')
savebestmodel = ModelCheckpoint(("results/model_" + mg_train + "X_K" + num_fold + "_" + str(width) + "x" + str(height) + ".h5"), save_best_only=True, monitor='val_loss', verbose=1, mode='min')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=reducelr_factor, patience=reducelr_patience, min_lr=min_learning_rate, verbose=1)
callback = [earlystopping, savebestmodel, reducelr]
hist = model.fit(generator, steps_per_epoch=len(x_train) / batch_size, epochs=num_epochs, verbose=1, callbacks=callback, validation_data=val_generator, validation_steps=len(x_val)/batch_size)

# create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Save accuracy / loss during training to pickle file
pickle.dump(hist.history, open(("results/history_" + mg_train + "X_K" + num_fold + "_" + str(width) + "x" + str(height) + ".pkl"), 'wb'))

print('history saved')
