# The range of magnification levels used
mgs = [20,24,28,32,36,40,50,60,70,80,90,100,116,133,150,166,183,200,233,266,300,333,366,400]
# The number of folds used during the K cross-validation
folds = [1,2,3,4,5]
# Patch height and width
width = 350
height = 230

# Network parameters
num_epochs = 50
learning_rate = 0.01
min_learning_rate = 0.001
momentum = 0.9
batch_size = 64
loss = "categorical_crossentropy"
metrics = ["accuracy"]
dropout_rate = 0.2
earlystopping_patience = 15
reducelr_patience = 5
reducelr_factor = 0.3
num_classes = 2

# Transfer Learning
weights = "imagenet"

# Data Augmentation
rescale = 1./255
horizontal_flip = True
vertical_flip = True

# Random seed
seed = 31415