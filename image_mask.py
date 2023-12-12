from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd

import torch
#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
#from PytorchWildlife.data import datasets as pw_data 
#from PytorchWildlife import utils as pw_utils
from moviepy.editor import VideoFileClip

def contains_animal(labels):
    for label in labels:
        if 'animal' in label:
            return True
    return False

"""def video_clip(im_file):
    # Open the video file
    vidcap = cv2.VideoCapture(im_file)

    # Get the frames per second
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_times = []
    success, image = vidcap.read()
    count = 0
    while success:
        # Save frame as image every second
        if count % fps == 0:
            frames.append(image)
            frame_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
            frame_times.append(frame_time)
        
        # Read the next frame
        success, image = vidcap.read()
        count += 1
    
    return frames, frame_times"""

# Open the video file
vidcap = cv2.VideoCapture('path_to_your_video.mp4')

# Calculate the reduced dimensions
scale_percent = 50  # percent of original size
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_percent / 100)
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_percent / 100)

frames = []
frame_times = []
while vidcap.isOpened():
    ret, frame = vidcap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Append the frame to the list
    frames.append(frame)

    # Get the time of the current frame
    frame_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
    frame_times.append(frame_time)

# Release the video file
vidcap.release()

#frames, frame_times = video_clip("C:\\Users\\tomoyakanno\\Documents\\CameraTraps-PytorchWildlife_Dev\\demo\\demo_data\\videos\\opossum_example.mp4")

# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% 
# Initializing the MegaDetectorV5 model for image detection
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)
transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                    stride=detection_model.STRIDE)

results = []

for i, f in enumerate(frame_times):
    print(f)
    frame = frames[i] # Get the frame
    frame = Image.fromarray(np.uint8(frames[i]))
    # Create a new image with the same size as the original image, and make it white (255)
    mask = Image.new('L', frame.size, 255)

    # Initialize ImageDraw
    draw = ImageDraw.Draw(mask)

    # Draw a black rectangle on the mask image
    draw.rectangle([      1132,         397,        1441,         679], fill=0)

    # Apply the mask to the original image
    img_with_mask = Image.composite(frame, Image.new('RGB', frame.size), mask)
    
    img = np.array(img_with_mask.convert("RGB"))

    result = detection_model.single_image_detection(transform(img), img.shape)

    results.append(result)

    if contains_animal(result['labels']):
        result['object'] = 1
    else:
        result['object'] = 0
    #result['img_id'] = f"opossum_example{i}.jpg".format(i)

results_dataframe = pd.DataFrame(results)
results_dataframe.to_csv("results.csv")

clip_times = []
start_time = None

for i, obj in enumerate(result['object']):
    if obj == 1 and (i == 0 or result['object'][i - 1] == 0):
        start_time = frame_times[i]
    elif obj == 0 and (i > 0 and result['object'][i - 1] == 1):
        end_time = frame_times[i - 1]
        clip_times.append((start_time, end_time))

# If the last object is 1, add the last clip
if result['object'][-1] == 1:
    clip_times.append((start_time, frame_times[-1]))


# Load the video
clip = VideoFileClip("path_to_your_video.mp4")

# Extract the clips and add them to a list
clips = [clip.subclip(start, end) for start, end in clip_times]

# For each time range in clip_times, extract that part and save it as a new video
for i, (start, end) in enumerate(clip_times):
    subclip = clip.subclip(start, end)
    subclip.write_videofile(f"output_{i}.mp4")

