import os

#Load data from Kaggle
os.environ['KAGGLE_USERNAME'] = "dchoi8"  # username from the json file
os.environ['KAGGLE_KEY'] = "9f037b1309436ac6b80ff27f3cb54b0f"  # key from the json file
import kaggle
api = kaggle.api()
api.authenticate()
api.dataset_download_files('rsrishav/youtube-trending-video-dataset', 'Kaggle Dataset', unzip=True)