from PIL import Image, ImageDraw
import cv2
import numpy as np

import torch
#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
#from PytorchWildlife.data import datasets as pw_data 
from PytorchWildlife import utils as pw_utils
#from moviepy.editor import VideoFileClip

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

for i, f in enumerate(frame_times):
    print(f)
    frame = frames[i] # Get the frame
    frame = Image.fromarray(np.uint8(frames[i]))
    
    
    img = np.array(frame.convert("RGB"))

    result = detection_model.single_image_detection(transform(img), img.shape)

    if contains_animal(result['labels']):
        pw_utils.save_detection_images(result, "output")
        print(result)
        break




