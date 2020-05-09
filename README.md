# AI Generated Google Doodles
Ever wanted to create your own Google Doodle? This project makes it possible. Just enter
the subject and event you want the Google to be about (ex. "Bill Gates" and "Birthday"),
and this program will create a custom Google Doodle for you.

<img src="Sample Doodles/Doodle6.png"/>

## What the Program Does
- Download background image for subject and event.
- Apply the style of the event background image to the content of the subject background image ("mash").
- Save an icon based on the subject.
- Classify the icon as a G, o, or l, so it can replace that letter in the "Google" text.
- Select a dominant colour from the icon.
- Create the "Google" text with the dominant colour and calculate the position of the icon.
- Overlay the "Google" text on the mashed background and place the icon.

## Usage
- Important - upload the "AI Google Doodle" folder to your Google Drive from here such that it has the following path: **My Drive/Colab Notebooks/AI Google Doodle/app**
- Upload the **Suhana_Nadeem_AI_Google_Doodle_Project.ipynb** to your Google Drive such that it has the following path: **My Drive/Suhana_Nadeem_AI_Google_Doodle_Project.ipynb**
- To open **Suhana_Nadeem_AI_Google_Doodle_Project.ipynb**, click "Open with" > "Connect more apps" > search for Google Colaboratory > "Connect". Then, open the project with Google Colaboratory. If you have used Colab before, you can just click "Open with Google Colaboratory".
- Install the dependencies by running the code blocks as described in the Google Colab project. Restart Colab's runtime by going to **Runtime > Restart runtime...** when the text comments I've left in the project specify to do so.
- **To Run:** run the last code block and click on the ngrok.io link generated to go to the web app. Enter the desired subject and event, click Generate AI Doodle, and see the results!
- You will able to see the program conducting the image downloads, mashing, etc. from the output of the last code block in Google Colab as it runs. The images downloaded (as well as final doodles) can be found in **My Drive/Colab Notebooks/AI Google Doodle/app/static/images**

## Notes
- For slow internet connections, change the numIterate variable in main.py to 500 or lower. You will have to reupload main.py in Google Drive as well.
- Unsure what to try? Enter a famous person as your subject and a famous painting as your event.
