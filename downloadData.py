"""
This is a helper file that was used to download data to train the icon classifier model.

Learned how to download images from Google from: Hardik Vasa - https://github.com/hardikvasa/google-images-download
"""

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()

# Specifying keywords and download limit to download multiple images.
searchKeywords= "male face icon png"
arguments = {"keywords":searchKeywords, "limit": 100, "print-urls": True,"f":"png","output_directory":"finalDataset\O"}
paths = response.download(arguments)
print(paths)

searchKeywords= "girl avatar icon png"
arguments = {"keywords":searchKeywords, "limit": 100, "print-urls": True,"f":"png","output_directory":"finalDataset\O"}
paths = response.download(arguments)
print(paths)

searchKeywords= "circle object icon png"
arguments = {"keywords":searchKeywords, "limit": 80, "print-urls": True,"f":"png","output_directory":"finalDataset\O"}
paths = response.download(arguments)
print(paths)
