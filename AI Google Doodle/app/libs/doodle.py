# Importing libraries needed for image download, manipulation, and classification.
from google_images_download import google_images_download
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from PIL import Image, ImageDraw, ImageFont
import random

# Importing neural style transfer class.
from libs.bgrMasher import BgrMasher

# Import the necessary packages for letter classification of icon.
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import imutils
import pickle
import os
from tensorflow import keras

"""
The Doodle class consists of methods to create and combine all the components of the Google
Doodle. This includes downloading needed images, resizing them, classifying the chosen icon
as G, l, or o, mashing the content and style images, and more, as described in main.py.
"""
class Doodle(object):

    """
    This method defines the attributes of the Doodle class. The subject and event chosen by the user
    are passed when Doodle is instantiated in Main.
    """
    def __init__(self,subjectName,eventName):
        self.response = google_images_download.googleimagesdownload()
        self.bgrMasher=BgrMasher(subjectName,eventName)
        self.subjectName=subjectName
        self.eventName=eventName
        self.clearBgrPath='/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/libs/clearBackground.png'
        self.iconSize=130
        self.doodlePath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.subjectName+self.eventName+"doodle.png"
    
    """
    Used to resize downloaded "content" and "style" images to required dimensions.
    """
    def resizeImg(self,img):
        # Set required dimensions.
        height, width = img.shape[:2]
        maxHeight = 405
        maxWidth = 720
        
        # Shrink and crop image if it is bigger than required.
        if maxHeight < height or maxWidth < width:
            # Find the scaling factor.
            scalingFactor = maxHeight / float(height)
            if maxWidth/float(width) < scalingFactor:
                scalingFactor = maxWidth / float(width)
            shrunkenImg = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
            resizedImg = shrunkenImg[:405, :720]

        # Enlarge and crop image 3x if it is significantly smaller than required.
        elif width < maxWidth/3.0 or height < maxHeight/3.0:
            enlargedImg = cv2.resize(img, (0,0), fx=3, fy=3)
            resizedImg = enlargedImg[0:405, 0:720]

        # Enlarge and crop image 2.5x if it is smaller than required.
        else:
            enlargedImg = cv2.resize(img, (0,0), fx=2.5, fy=2.5)
            resizedImg = enlargedImg[0:405, 0:720]

        return resizedImg

    """
    Save the subject background - used as the content of the final background.
    Learned how to download images from Google from: Hardik Vasa - https://github.com/hardikvasa/google-images-download
    """
    def saveSubjectBackground(self):
        # Specifying keywords and download limit to download multiple subject backgrounds.
        searchKeywords=self.subjectName+" desktop wallpaper"
        downloadLimit=10
        arguments={"keywords":searchKeywords, "limit": downloadLimit}
        paths=self.response.download(arguments)
        
        # If any of the images were not successfully downloaded, they must be re-downloaded.
        while(not paths[0][searchKeywords] or not paths[0][searchKeywords][0]):
            paths = self.response.download(arguments)

        # Choosing and reading random image from those downloaded.
        randInt = random.randint(0,downloadLimit-1)
        path = paths[0][searchKeywords][randInt]
        subjectImg = cv2.imread(path)

        # Re-reading the image in case it did not load successfully (if cv2.imread returned None).
        while (subjectImg is None):
            subjectImg = cv2.imread(path)

        # Save resized background image with appropriate path. 
        resizedSubjectImg = self.resizeImg(subjectImg)
        subjectBgrPath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.subjectName+self.eventName+"subjectBackground.png"
        cv2.imwrite(subjectBgrPath,resizedSubjectImg)
        return subjectBgrPath

    """
    Save the event background - used as the style of the final background.
    Learned how to download images from Google from: Hardik Vasa - https://github.com/hardikvasa/google-images-download
    """
    def saveEventBackground(self):
        # Specifying keywords and downloading one event background.
        searchKeywords=self.eventName+" wallpaper"
        arguments={"keywords":searchKeywords, "limit": 1}
        paths=self.response.download(arguments)
        
        # If the image was not successfully downloaded, it must be re-downloaded.
        while(not paths[0][searchKeywords] or not paths[0][searchKeywords][0]):
            paths = self.response.download(arguments)

        # Reading downloaded image.
        path = paths[0][searchKeywords][0]
        eventImg = cv2.imread(path)

        # Re-reading the image in case it did not load successfully (if cv2.imread returned None).
        while (eventImg is None):
            eventImg = cv2.imread(path)

        # Save resized background image with appropriate path. 
        resizedEventImg = self.resizeImg(eventImg)
        eventBgrPath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.subjectName+self.eventName+"eventBackground.png"
        cv2.imwrite(eventBgrPath,resizedEventImg)
        return eventBgrPath

    """
    Apply the style of the event background to the content of the subject background.
    """
    def saveMashedBackground(self,subjectBgrPath,eventBgrPath,numIterate):
        contentPath = subjectBgrPath
        stylePath = eventBgrPath

        # Call the transferStyle method on the bgrMasher object and return mashed background.
        mashedBgrPath=self.bgrMasher.transferStyle(contentPath=contentPath,
                            stylePath=stylePath, numIterations=numIterate)
        
        return mashedBgrPath

    def saveIcon(self):
        # Specifying keywords and download limit to download multiple icons.
        searchKeywords=self.subjectName+" png icon transparent"
        downloadLimit=15
        # Only download png images.
        arguments={"keywords":searchKeywords, "limit": downloadLimit, "f":"png"}
        paths=self.response.download(arguments)
        
        # If the image was not successfully downloaded, it must be re-downloaded.
        while(not paths[0][searchKeywords] or not paths[0][searchKeywords][0]):
            paths = self.response.download(arguments)
        
        # Reading each downloaded image until a png image with a transparent background
        # is successfully downloaded.
        path = paths[0][searchKeywords][0]
        iconImg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        iconIndex=1
        while(iconIndex<downloadLimit and (iconImg is None or iconImg.shape is None or len(iconImg.shape)<3 or iconImg.shape[2]<4)):
            path = paths[0][searchKeywords][iconIndex]
            iconImg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            iconIndex+=1

        # Learned how to remove non-transparent part of png from: https://stackoverflow.com/questions/46273309/using-opencv-how-to-remove-non-object-in-transparent-image
        
        # Find min and max x and y values from all non-zero alpha coordinates in the image.
        imgY,imgX = iconImg[:,:,3].nonzero()
        minX = np.min(imgX)
        minY = np.min(imgY)
        maxX = np.max(imgX)
        maxY = np.max(imgY) 

        # Crop the image to exclude excess alpha area, using minY and maxY for height
        # and minX and maxX for width.
        croppedIconImg = iconImg[minY:maxY, minX:maxX]

        height, width = croppedIconImg.shape[:2]
        maxHeight = self.iconSize
        maxWidth = maxHeight
        
        # Shrink the icon while preserving aspect ratio if it's bigger than required.
        if maxHeight < height or maxWidth < width:
            # Find the scaling factor.
            scalingFactor = maxHeight / float(height)
            if maxWidth/float(width) < scalingFactor:
                scalingFactor = maxWidth / float(width)
            # Resize the icon.
            resizedIconImg = cv2.resize(croppedIconImg, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)

        # Save the icon.
        iconPath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.subjectName+self.eventName+"subjectIcon.png"
        cv2.imwrite(iconPath,resizedIconImg)
        return iconPath

    """
    This method gets the dominant colour in an image. It uses K-Means Clustering, an unsupervised
    learning method for data samples that do not have labels. It finds K number of groups in the data
    and assigns each data group to one of the K groups based on their features.
    Learned how to get the dominant colours from an image from: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
    """ 
    def getDominantColour(self,iconPath):
        # Read icon image and recolour to RGB colour channel.
        img = cv2.imread(iconPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a 2D Matrix: row*column, channel number.
        img = img.reshape((img.shape[0] * img.shape[1],3))
        # Applying K-Means to form 3 clusters. 
        clt = KMeans(n_clusters=3)
        clt.fit(img)
        histogram = self.getHistogram(clt)

        # Top three dominant colours.
        dominantColours=[[0,0,0],[0,0,0],[0,0,0]]

        # Add least and most dominant colour to list using the histogram; 
        # the 0th element is least dominant and 2nd is most.
        highestPercentage=histogram[0]
        lowestPercentage=histogram[0]
        for (percent, colour) in zip(histogram, clt.cluster_centers_):
            if (percent>=highestPercentage):
                highestPercentage=percent
                dominantColours[2]=colour
            if (percent<lowestPercentage):
                lowestPercentage=percent
                dominantColours[0]=colour

        # Find the second most dominant colour and add to list as 1st element.
        for (percent, colour) in zip(histogram, clt.cluster_centers_):
            if (lowestPercentage<percent and highestPercentage>percent):
                dominantColours[1]=colour
        
        return dominantColours

    """
    This method is used to create a histogram to find the dominant colour 
    of an image using K-Means clustering. The histogram returned here is a
    graphical representation of the distribution of the three most dominant
    colours throughout the image.
    """
    def getHistogram(self,clt):
        # Calculating histogram values for each pixel.
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (histogram, _) = np.histogram(clt.labels_, bins=numLabels)

        # Calculating histogram ratios.
        histogram = histogram.astype("float")
        histogram /= histogram.sum()

        return histogram

    """
    This method is used to utilize a model (trained by me) to classify the choosen icon as a G, l, or o.
    Training and classifying using a CNN was learned from: https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
    """
    def classifyIcon(self, model, labelBinarizer, iconPath):
        # Load the icon, convert to black and white, and pre-process for classification.
        iconImg = cv2.imread(iconPath)
        iconImg = cv2.cvtColor(iconImg, cv2.COLOR_BGR2GRAY)
        iconImg = cv2.resize(iconImg, (96, 96))
        iconImg = iconImg.astype("float") / 255.0
        iconImg = img_to_array(iconImg)
        iconImg = np.expand_dims(iconImg, axis=0)

        # Load the trained CNN and the label binarizer.
        model = keras.models.load_model(model)
        loadedBinarizer = pickle.loads(open(labelBinarizer, "rb").read())

        # Classify the icon - choose the class that returned the highest probability.
        probability = model.predict(iconImg)[0]
        index = np.argmax(probability)
        label = loadedBinarizer.classes_[index] # Convert label so it's human-readable.

        return label

    """
    This method uses the dominant colour chosen to return the "Google" text component of the doodle, along
    with the position the icon must be placed (according to which letter it was matched to).
    """
    def makeGoogleText(self,dominantColours, matchedLetter):
        # Declare fonts to be used and choose a random one from the list.
        fonts = ['SHOWG.ttf', 'ARIALBD.ttf','BAUHS93.ttf', 'CALIBRIB.ttf', 'SWISSK.ttf','COOPBL.ttf']
        i = random.randint(0,len(fonts)-1)
        fontPath = '/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/fonts/'+fonts[i]

        # Load and initialize the clear image as the base layer of the Doodle.
        clearBackground = Image.open(self.clearBgrPath)
        draw = ImageDraw.Draw(clearBackground)

        # Load chosen font in needed size.
        fontSize=130
        iconGap=2
        font=ImageFont.truetype(fontPath, size=fontSize)

        # Get the RGB colour of the least dominant colour of the icon.
        dominanceRank=0
        redVal,greenVal,blueVal="0","0","0"
        redVal=str(int(dominantColours[dominanceRank][0]))
        greenVal=str(int(dominantColours[dominanceRank][1]))
        blueVal=str(int(dominantColours[dominanceRank][2]))
        colour = 'rgb('+redVal+','+greenVal+','+blueVal+')'
        
        # Define the message (letters written before icon) and leftover message (letters 
        # written after the icon) based on which icon was chosen.
        if matchedLetter == "G":
            message = ""
            leftoverMessage = "OOGLE"
        elif matchedLetter == "l":
            message = "GOOG"
            leftoverMessage = "E"   
        else:
            message = "GO"
            leftoverMessage = "GLE"  

        # Use the height and width of the font to calculate centered coordinates for "Google" and draw it
        # on the clear background.
        imgX = ((720)-(self.iconSize+iconGap*2+font.getsize(message)[0]+font.getsize(leftoverMessage)[0]))/2.0
        imgY = (405-font.getsize(leftoverMessage)[1])/2.0
        draw.text((imgX, imgY), message, fill=colour, font=font)
        
        # Calculate the placement of the icon.
        iconPositionX=imgX+font.getsize(message)[0]+iconGap
        iconPositionY=(405-self.iconSize)/2.0
            
        # Calculate the imgX-coordinate of the leftover message and draw it.
        imgX=iconPositionX+iconGap+self.iconSize
        draw.text((imgX, imgY), leftoverMessage, fill=colour, font=font)

        # Save the "Google" text in the appropriate location.
        googleTextPath="/content/drive/My Drive/Colab Notebooks/AI Google Doodle/app/static/images/"+self.subjectName+self.eventName+'googleText.png'
        clearBackground.save(googleTextPath)
        return googleTextPath,iconPositionX,iconPositionY

    """
    This method overlays all the components of the final Google Doodle on each other. This is like "pasting" 
    them on top of each other, and is achieved using PIL.
    """
    def makeFinalDoodle(self,mashedBackgroundPath,iconPath,googleTextPath,iconPositionX,iconPositionY):
        background = Image.open(self.clearBgrPath)

        # To make sure the path is opened, properly.
        mashedBackground = cv2.imread(mashedBackgroundPath,-1)
        while (mashedBackground is None):
            mashedBackground = cv2.imread(mashedBackgroundPath,-1)
        
        # Paste the mashed background on the clear background.
        mashedBackground = Image.open(mashedBackgroundPath)
        background.paste(mashedBackground, (0, 0),mashedBackground.convert('RGBA'))

        # Paste the google text on the mashed background.
        googleText=Image.open(googleTextPath)
        background.paste(googleText, (0, 0),googleText.convert('RGBA'))

        # Adjust the imgX position of the icon if it is narrow in height or width.
        iconImg=Image.open(iconPath)
        iconWidth, iconHeight = iconImg.size
        if iconWidth < 60:
            iconPositionX += 50
        elif iconWidth < 90:
            iconPositionX += 30
        if iconHeight < 80:
            iconPositionY += 50
        
        # Paste the icon on the google text.
        background.paste(iconImg, (int(iconPositionX), int(iconPositionY)),iconImg.convert('RGBA'))

        background.save(self.doodlePath)
        return self.doodlePath