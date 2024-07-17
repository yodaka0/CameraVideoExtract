# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for image detection"""

#%% 
# Importing necessary basic libraries and modules
import numpy as np
import os
from PIL import Image
import cv2
import shutil

#%% 
# PyTorch imports 
import torch
#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection


def video_clip(im_file):
    cliped_frames_path = None
    # make a directory to store the frames
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    for ext in video_extensions:
        if im_file.lower().endswith(ext):
            ext_start_index = im_file.lower().rfind(ext.lower())
            cliped_frames_path = im_file[:ext_start_index] + im_file[ext_start_index+len(ext):]
    #print(cliped_frames_path)
    if not os.path.exists(cliped_frames_path):
        os.makedirs(cliped_frames_path)
    # Open the video file
    vidcap = cv2.VideoCapture(im_file)

    # Get the frames per second
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    frames = []
    success, image = vidcap.read()
    count = 0
    while success:
        # Save frame as image every second
        if count % fps == 0:
            frames.append(image)
            # Save the frame as an image
            cv2.imwrite(cliped_frames_path + "\\frame%d.jpg" % count, image)
        
        # Read the next frame
        success, image = vidcap.read()
        count += 1
    
    return cliped_frames_path, count

def pw_detect(im_file, new_file, threshold=None):
    if not isinstance(threshold, float):
        threshold = 0.2

    #%% 
    # Setting the device to use for computations ('cuda' indicates GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    #%% 
    # Initializing the MegaDetectorV6 model for image detection
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, weights='../MDV6b-yolov9c.pt', pretrained=False)

    cliped_frames_path, count = video_clip(im_file)

    #%% Single image detection
    #object = list()

    results = detection_model.batch_image_detection(cliped_frames_path, batch_size=count, conf_thres=threshold)

    animal_ns = []
    first = True
    for result in results:
        animal_ns.append(sum('animal' in item for item in result['labels']))
        #save first result
        if first:
            first = False
            result_first = result
        #print(animal_ns[-1])
        if animal_ns[-1] > 0:
            #print('Animal detected')
            # copy the video to the new file
            shutil.copy(im_file, new_file)

    result_first['animal_ns'] = animal_ns
    
    return result_first

    
