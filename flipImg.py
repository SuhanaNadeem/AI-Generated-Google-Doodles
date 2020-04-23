"""
This is a helper file that was used to go through each image in a folder and flip it
vertically, for the "G" training images of the icon classifier (the model was trained so
that only icons facing right should be classified as "G"s, and so right-facing icons 
needed to be flipped).

Learned how to iterate through a folder and flip images from Python and numpy docs.
"""

from scipy import ndimage, misc
import numpy as np
import os
import cv2

outputFolderPath = "" # path where final image was saved went here.
inputFolderPath = "" # path where original image was downloaded went here.

for imgPath in os.listdir(inputFolderPath):
    # Load the image.
    inputImgPath = os.inputFolderPath.join(inputFolderPath, imgPath)
    img = ndimage.imread(inputImgPath)

    # Flip the image.
    flippedImg = np.fliplr(img)

    # Save the image.
    outputImgPath = os.inputFolderPath.join(outputFolderPath, 'flipped'+imgPath)
    misc.imsave(outputImgPath, flippedImg)