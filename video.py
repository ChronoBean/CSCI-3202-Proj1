# Uncomment the line below and run this only if you're using Google Collab
# !pip install -q py-feat
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from feat import Detector
from feat.utils.io import get_test_data_path
import os
from IPython.core.display import Video

# Setting up the Detector
detector = Detector()

# User interface
file_name = input("Enter the name of your file (must be of type mp4 in the downloads folder): ")

# Processing videos
test_data_dir = get_test_data_path()
test_video_path = os.path.join(os.path.expanduser("~"), "Downloads", file_name + ".mp4")

# Show video
Video(url=test_video_path, embed=True)

# Detecting facial expressions in videos
video_prediction = detector.detect_video(test_video_path, skip_frames=24)
print(video_prediction.head())

# Visualizing predictions
start = int(input("Enter the bounds of where detection should start (should be within the frame of the mp4): "))
finish = int(input("Enter the bounds of where detection should end (should be within the frame of the mp4 & greater than your last entry): "))
video_prediction.loc[[start, finish]].plot_detections(faceboxes=False, add_titles=False)

# Plot emotions over time
axes = video_prediction.emotions.plot()

# ONLY UNCOMMENT THE LINE BELOW IF YOU WANT TO Run prediction for every video frame (may take a VERY long while)
# video_prediction = detector.detect_video(test_video_path)
print("Full emotion spread by frame intervals:")
print(video_prediction.emotions)
