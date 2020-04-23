"""
This file is used to apply Neural Style transfer on the subject and event background. This 
is an optimization technique used to take a content image and a style image, and mash them
together so the input image is transformed to have the content of the content image but the
style of the style image.

This technique and this code was learned and adapted from the following interpretations of Neural Style Transfer:
- Leon A. Gatys' research paper: https://arxiv.org/abs/1508.06576
- Style Transfer using Eager Execution: https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb

Changes Made to Source Code:
- Optimization process was changed to include an Adam optimizer that is more compatible 
with current versions of TensorFlow.
- The process was adapted for use in Google Colab.
- The process was made object oriented.
- The final mashed image was made to be saved to Google Drive.
"""

# Import the necessary packages for plotting intermediate images.
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

# Import packages for image manipulation.
import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf

# Import packages to be used for preprocessing images with keras.
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import IPython.display

# Import packages for Google Drive authentication.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client to be able to upload images to Drive.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()

"""
We need both the content and style representations of our image, so we need to access
some intermediate layers within our model
"""
# Content layer from which feature maps will be pulled.
contentLayers = ['block5_conv2'] 
# Style layers needed.
styleLayers = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1' ]

numContentLayers = len(contentLayers)
numStyleLayers = len(styleLayers)

"""
This class defines all the methods needed to preprocess/deprocess images, compute losses, and 
get image feature representations to perform Neural Style Transfer.
"""
class BgrMasher(object):

    """
    Defining the path of the final mashed image.
    """
    def __init__(self,subjectName,eventName):
        self.mashedImgName = subjectName+eventName+"mashedBackground.png"
        self.mashedBackgroundPath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.mashedImgName

    """
    Method used to load, scale, and convert images to .
    """
    def loadImg(self, pathToImg):
        maxDim = 720
        img = Image.open(pathToImg)
        long = max(img.size)
        scale = maxDim/long
        resizedImg = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
        
        convertedImg = kp_image.img_to_array(resizedImg)

        # Broadcast the image array so it has a batch dimension.
        loadedImg = np.expand_dims(convertedImg, axis=0)
        return loadedImg
    
    """
    This method prepares an image for style transfer as outlined in the VGG19 model's
    training process. VGG networks are trained on images with each BGR channel normalized by 
    mean = [103.939, 116.779, 123.68].
    """
    def loadAndProcessImg(self, pathToImg):
        img = self.loadImg(pathToImg)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    """
    This method deprocesses am image so we can see the outputs of our optimization. This
    involves performing the inverse of the preprocessing step above.
    """    
    def deprocessImg(self, preprocessedImg):
        # Ensure the image has the needed shape.
        deprocessedImg = preprocessedImg.copy()
        if len(deprocessedImg.shape) == 4:
            deprocessedImg = np.squeeze(deprocessedImg, 0)
        assert len(deprocessedImg.shape) == 3, ("Input to deprocess image must be an image of "
                            "dimension [1, height, width, channel] or [height, width, channel]")
        if len(deprocessedImg.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # Perform the inverse of preprocessing.
        deprocessedImg[:, :, 0] += 103.939
        deprocessedImg[:, :, 1] += 116.779
        deprocessedImg[:, :, 2] += 123.68
        deprocessedImg = deprocessedImg[:, :, ::-1]

        # Maintain values within the 0-255 range for each channel.
        deprocessedImg = np.clip(deprocessedImg, 0, 255).astype('uint8')
        return deprocessedImg

    """
    This method will use the VGG19 model to access the intermediate layers, in order to 
    make a new model that can take an input image and return the outputs from these 
    intermediate layers of VGG19. The method returns a keras model that provides the
    style and content intermediate layers of an input image. 
    """
    def getModel(self):
        # Load pretrained VGG19 that has been trained on ImageNet data.
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Access the outputs of the style and content layers.
        styleOutputs = [vgg.get_layer(name).output for name in styleLayers]
        contentOutputs = [vgg.get_layer(name).output for name in contentLayers]
        modelOutputs = styleOutputs + contentOutputs
        return models.Model(vgg.input, modelOutputs)

    """
    This method is used to calculate content loss between the base input image and the 
    desired content image. It takes these two images and returns the euclidean 
    distance between the two intermediate representations of those images.
    """
    def getContentLoss(self, baseContent, target):
        return tf.reduce_mean(tf.square(baseContent - target))

    """
    Similar to what we did for content loss, we use the following two methods to calculate 
    the style loss. We compare the Gram matrices of the two outputs of the base input image 
    and the style image passed into VGG instead of the raw intermediate outputs. 
    """
    def getGramMatrix(self, inputTensor):
        # Get the image channels.
        channels = int(inputTensor.shape[-1])
        a = tf.reshape(inputTensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def getStyleLoss(self, baseStyle, gramTarget):
        # Scale the loss at a given layer by the size of the feature map and the # of filters.
        height, width, channels = baseStyle.get_shape().as_list()
        gramStyle = self.getGramMatrix(baseStyle)

        return tf.reduce_mean(tf.square(gramStyle - gramTarget))

    """
    Compute content and style feature representations. This method loads and preprocesses 
    the content and style images, and passes them to the network to obtain the outputs of 
    the intermediate layers. 
    """
    def getFeatureRepresentations(self, model, contentPath, stylePath):
        # Load images.
        contentImage = self.loadAndProcessImg(contentPath)
        styleImage = self.loadAndProcessImg(stylePath)

        # Compute content and style features.
        styleOutputs = model(styleImage)
        contentOutputs = model(contentImage)

        # Get the style and content feature representations from the outputs of the model.
        styleFeatures = [styleLayer[0] for styleLayer in styleOutputs[:numStyleLayers]]
        contentFeatures = [contentLayer[0] for contentLayer in contentOutputs[numStyleLayers:]]
        return styleFeatures, contentFeatures

    """
    This method computes the total loss.

    Arguments:
    model: The model that will give us access to the intermediate layers
    lossWeights: The weights of each contribution of each loss function. 
    (style weight, content weight, and total variation weight)
    initImage: Our initial base image, which is being updated with our 
    optimization process. We apply the gradients wrt the loss we are 
    calculating to this image.
    gramStyleFeatures: Precomputed gram matrices corresponding to the 
    defined style layers of interest.
    contentFeatures: Precomputed outputs from defined content layers of 
    interest.

    Returns:
    returns the total loss, style loss, content loss, and total variational loss.
    """
    def computeLoss(self, model, lossWeights, initImage, gramStyleFeatures, contentFeatures):
        styleWeight, contentWeight = lossWeights

        # Pass init image through model to get the content and style representations 
        # at the desired layers. Model is callable since Eager Execution is used.
        modelOutputs = model(initImage)

        styleOutputFeatures = modelOutputs[:numStyleLayers]
        contentOutputFeatures = modelOutputs[numStyleLayers:]

        styleScore = 0
        contentScore = 0

        # Accumulate style losses from all layers; equally weight each contribution of each loss layer.
        weightPerStyleLayer = 1.0 / float(numStyleLayers)
        for targetStyle, combStyle in zip(gramStyleFeatures, styleOutputFeatures):
            styleScore += weightPerStyleLayer * self.getStyleLoss(combStyle[0], targetStyle)

        # Accumulate content losses from all layers. 
        weightPerContentLayer = 1.0 / float(numContentLayers)
        for targetContent, combContent in zip(contentFeatures, contentOutputFeatures):
            contentScore += weightPerContentLayer* self.getContentLoss(combContent[0], targetContent)

        styleScore *= styleWeight
        contentScore *= contentWeight

        # Get total loss.
        loss = styleScore + contentScore 
        return loss, styleScore, contentScore

    """
    This method uses tf.GradientTape to compute the gradient of the loss function with
    respect to the input image.
    """
    def computeGrads(self, cfg):
        # Compute total loss.
        with tf.GradientTape() as tape: 
            allLoss = self.computeLoss(**cfg)
        totalLoss = allLoss[0]
        return tape.gradient(totalLoss, cfg['initImage']), allLoss

    """
    This method contains the optimization loop, to iteratively make the content and the
    style of the base image as accurate as possible.
    """
    def transferStyle(self, contentPath, 
                        stylePath,
                        numIterations=1000,
                        contentWeight=1e3, 
                        styleWeight=1e-2): 
        # We don't need to train any layers of our model.       
        model = self.getModel() 
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (based on the specified intermediate layers).
        styleFeatures, contentFeatures = self.getFeatureRepresentations(model, contentPath, stylePath)
        gramStyleFeatures = [self.getGramMatrix(styleFeature) for styleFeature in styleFeatures]

        # Set initial image.
        initImage = self.loadAndProcessImg(contentPath)
        initImage = tf.Variable(initImage, dtype=tf.float32)
        # Create the optimizer.
        optimizer = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

        # For displaying intermediate images while style transfer is occurring.
        iterCount = 1

        # Store the best result.
        bestLoss, bestImg = float('inf'), None

        # Create a configuration. 
        lossWeights = (styleWeight, contentWeight)
        cfg = {
            'model': model,
            'lossWeights': lossWeights,
            'initImage': initImage,
            'gramStyleFeatures': gramStyleFeatures,
            'contentFeatures': contentFeatures
        }

        # For displaying the intermediate images.
        numRows = 2
        numCols = 5
        displayInterval = numIterations/(numRows*numCols)
        startTime = time.time()
        globalStart = time.time()

        # Define normalized means, and minimum/maximum values for BGR channels.
        normalizedMeans = np.array([103.939, 116.779, 123.68])
        minVals = -normalizedMeans
        maxVals = 255 - normalizedMeans   

        # Continue to compute loss and iteratively compute/apply gradients until we reach
        # the optimal loss.
        imgs = []
        for i in range(numIterations):
            print(i)
            gradients, allLoss = self.computeGrads(cfg)
            loss, styleScore, contentScore = allLoss
            optimizer.apply_gradients([(gradients, initImage)])
            clipped = tf.clip_by_value(initImage, minVals, maxVals)
            initImage.assign(clipped)
            endTime = time.time() 
            
            if loss < bestLoss:
                # Update best loss and best image from total loss. 
                bestLoss = loss
                bestImg = self.deprocessImg(initImage.numpy())

            if i % displayInterval==0:
                startTime = time.time()
                
                # Use the .numpy() method to get the concrete numpy array.
                plotImg = initImage.numpy()
                plotImg = self.deprocessImg(plotImg)
                imgs.append(plotImg)
                IPython.display.clear_output(wait=True)
                IPython.display.display_png(Image.fromarray(plotImg))
                print('Iteration: {}'.format(i))        
                print('Total loss: {:.4e}, ' 
                        'style loss: {:.4e}, '
                        'content loss: {:.4e}, '
                        'time: {:.4f}s'.format(loss, styleScore, contentScore, time.time() - startTime))
       
        # Display each iteration's image at the end.
        print('Total time: {:.4f}s'.format(time.time() - globalStart))
        IPython.display.clear_output(wait=True)
        plt.figure(figsize=(14,4))
        for i,img in enumerate(imgs):
            plt.subplot(numRows,numCols,i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        # Save the final image.
        mashed = Image.fromarray(bestImg)
        mashed.save(self.mashedBackgroundPath)

        return self.mashedBackgroundPath