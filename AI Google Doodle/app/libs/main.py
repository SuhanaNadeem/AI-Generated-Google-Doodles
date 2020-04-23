# Must import the Doodle class to access its methods.
from libs.doodle import Doodle

"""
The Main class is instatiated by Flask through Google Colab. Its getDoodle() method calls the 
needed methods from doodle.py to generate the Google Doodle in these major steps:
1. Download background image for subject and event.
2. Apply the style of the event background image to the content of the subject background image ("mash").
3. Save an icon based on the subject.
4. Classify the icon as a G, o, or l, so it can replace that letter in the "Google" text.
5. Select a dominant colour from the icon.
6. Create the "Google" text with the dominant colour and calculate the position of the icon.
7. Overlay the "Google" text on the mashed background and place the icon.
"""

class Main(object):

    """
    The parameters referenced here are entered by the user through the website, and 
    passed during the instantiation of Main through Colab.
    """
    def __init__(self,subjectName,eventName):
        self.subjectName=subjectName
        self.eventName=eventName
        # Create a Doodle object to access the functionalities needed to create the 
        # different components of and assemble the doodle.
        self.doodleMaker=Doodle(self.subjectName,self.eventName)
    
    """
    getDoodle() calls the methods needed on the doodleMaker object in order to save
    subject and event backgrounds from Google, create the mashed background, save and 
    classify the icon, and assemble all these components for the find doodle.
    """
    def getDoodle(self):
        # Saving the subject and background images and getting their paths.
        subjectBgrPath=self.doodleMaker.saveSubjectBackground()
        eventBgrPath=self.doodleMaker.saveEventBackground()      
        
        # Applying the style of the event image to the content of the background image to create
        # the background. 1000 iterations are used for best results.
        numIterate=1000
        mashedBackgroundPath=self.doodleMaker.saveMashedBackground(subjectBgrPath,eventBgrPath,numIterate)

        # Saving the icon and classifying it as a G, o, or l.
        iconPath=self.doodleMaker.saveIcon()
        matchedLetter=self.doodleMaker.classifyIcon("//content//drive//My Drive//Colab Notebooks//AI Google Doodle//app//libs//iconClassification.model", "//content//drive//My Drive//Colab Notebooks//AI Google Doodle//app//libs//labelbin.pickle", iconPath)

        # Get the dominant colours from the icon and make the "Google" word art using one of those colours.
        dominantColours=self.doodleMaker.getDominantColour(iconPath)
        googleTextPath,iconPositionX,iconPositionY=self.doodleMaker.makeGoogleText(dominantColours,matchedLetter)

        # Assemble the final Google Doodle by overlaying all the components.
        finalImagePath=self.doodleMaker.makeFinalDoodle(mashedBackgroundPath,iconPath,googleTextPath,iconPositionX, iconPositionY)