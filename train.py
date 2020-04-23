"""
This file was used to train the model (iconClassification.model) that classifies icons as G, l, or o.
The library "pyimagesearch" is used to access a small VGG for these training purposes.

To train your own model using my dataset, run:
- python train.py --dataset finalDataset --model iconClassification.model --labelbin labelbin.pickle

Image classification was learned and this training file was adapted from:
- Adrian Roseblock: https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

Major Changes Made to Source Code:
- The training configurations were adjusted to be better suited for my dataset. The epochs,
learning rate, and batch sizes were optimized. 
- The training data was created by me, so the model this file trains is unique.
"""

# Import the necessary packages.
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Parse arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Declare the number of epochs to train for, the initial learning rate,
# batch size, and image dimensions.
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 1)

data = []
labels = []

# Load the image paths and randomly shuffle them.
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(4)
random.shuffle(imagePaths)

# Iterate over the image paths.
for imagePath in imagePaths:
	# Load the image, pre-process it, and store it in the data list.
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# Extract the corresponding class label from the image path and update the
	# labels list.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# Scale the raw pixel intensities to the range [0, 1].
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# Binarize the labels - transform class labels into integer label 
# predictions to be used for training.
labelbin = LabelBinarizer()
labels = labelbin.fit_transform(labels)

# Split the data into training and testing data using 80% of
# the data for training and 20% for testing.
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=4)

# Construct the image generator for data augmentation - the 
# ImageDataGenerator applies a series of random transformations
# to each image in the training data, so the network sees "new"
# images in each epoch.
augmentation = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# Create the model and optimize using Adam.
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=3)
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
	metrics=["accuracy"])

# Train the network.
print("[INFO] training network...")
H = model.fit_generator(
	augmentation.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# Save the model.
print("[INFO] serializing network...")
model.save(args["model"])

# Save the label binarizer.
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(labelbin))
f.close()