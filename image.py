# Uncomment the line below and run this only if you're using Google Collab
# !pip install -q py-feat
# Otherwise just run the following command:
# pip install py-feat
import os
from feat import Detector
from feat.utils.io import get_test_data_path, read_feat
from feat.plotting import imshow

# Setting up the Detector
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

# Helper to point to the test data folder
test_data_dir = get_test_data_path()

# Get the full path of the single face image
file_name = input("Enter the name of your file (must be of type jpg in the downloads folder): ")

# Determine whether or not there are one or multiple faces in the image
m_faces = input("Are there more faces than 1 in your image? (y/n): ")
if m_faces == "y":
    # Detect multiple faces from a single image
    multi_face_image_path = os.path.join(os.path.expanduser("~"), "Downloads", file_name + ".jpg")
    multi_face_prediction = detector.detect_image(multi_face_image_path)

    # Show results
    print(multi_face_prediction)

    # Plot detection results and poses
    figs = multi_face_prediction.plot_detections(add_titles=False)

    # Detect faces from multiple images
    img_list = [multi_face_image_path]
    mixed_prediction = detector.detect_image(img_list, batch_size=1)
    print(mixed_prediction)

    # Plot detection results for all images
    figs = mixed_prediction.plot_detections(add_titles=False)

    # Plot detection for a specific image
    img_name = mixed_prediction['input'].unique()[0]
    axes = mixed_prediction.query("input == @img_name").plot_detections(add_titles=False)

    print("Full Emotion Spread: (The different people will be numbered 0-X from Left-Right)")
    print(multi_face_prediction.emotions)

else:
    #Path to the single face image
    single_face_img_path = os.path.join(os.path.expanduser("~"), "Downloads", file_name + ".jpg")

    # Plot the single face image
    imshow(single_face_img_path)

    # Process a single image with a single face
    single_face_prediction = detector.detect_image(single_face_img_path)

    # Show results
    print(single_face_prediction)

    # Access columns of interest
    print(single_face_prediction.facebox)
    print(single_face_prediction.aus)
    print(single_face_prediction.emotions)
    print(single_face_prediction.facepose)

    # Save detection to a file
    single_face_prediction.to_csv("output.csv", index=False)

    # Load detection results from a saved file
    input_prediction = read_feat("output.csv")

    # Show loaded results
    print(input_prediction)

    # Visualize detection results
    figs = single_face_prediction.plot_detections(poses=True)

    # Format the results
    print("Full emotion spread:")
    print(single_face_prediction.emotions)
