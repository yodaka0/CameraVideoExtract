
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
#from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

#from classifier import Classifier

def video_clip(im_file, num_segments_per_sec=1):
    # Create a folder to store the frames
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    cliped_frames_path = None
    for ext in video_extensions:
        if im_file.lower().endswith(ext):
            ext_start_index = im_file.lower().rfind(ext.lower())
            cliped_frames_path = im_file[:ext_start_index] + im_file[ext_start_index+len(ext):]
            break  # once found, exit the loop

    if cliped_frames_path is None:
        raise ValueError("Unsupported video extension")
    else:
        if not os.path.exists(cliped_frames_path):
            os.makedirs(cliped_frames_path)

    # Extract metadata using hachoir
    parser = createParser(im_file)
    if not parser:
        print("Unable to parse file")
        metadata = None
    else:
        with parser:
            metadata = extractMetadata(parser)
            if not metadata:
                print("Unable to extract metadata")
                metadata = None
            else:
                metadata = metadata.exportDictionary()

    # Open the video file
    vidcap = cv2.VideoCapture(im_file)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_meta = metadata[36868] - metadata[36867]
    if video_length_meta != total_frames:
        print(f"Video length in metadata ({video_length_meta}) does not match the actual number of frames ({total_frames})")
    
    # Compute the frame indices to sample (evenly spaced)
    if num_segments_per_sec < 1:
        raise ValueError("num_segments must be at least 1")
    indices = np.linspace(0, total_frames - 1, num_segments_per_sec*total_frames, dtype=int)

    frames = []
    create_img_pathes = []
    for idx in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = vidcap.read()
        if not success:
            continue
        frames.append(image)
        frame_filename = os.path.join(cliped_frames_path, f"frame{idx}.jpg")
        create_img_pathes.append(frame_filename)
        cv2.imwrite(frame_filename, image)

    return cliped_frames_path, len(frames), metadata, create_img_pathes


def pw_detect(im_file, new_file, threshold=None, video_sep=1, verbose=False, model=None, dir_remove=1):

    if not isinstance(threshold, float):
        threshold = 0.2

    classify = False
    
    #%% 
    # Setting the device to use for computations ('cuda' indicates GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(DEVICE)
        print(f"Threshold: {threshold}")
    #%% 
    # Initializing the MegaDetectorV5 model for image detection
    if model == "MegaDetector_v5":
        detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")
    elif model == "HerdNet":
        if DEVICE == "cpu":
            print("HerdNet model is too heavy for CPU, please use GPU")
            raise ValueError("Model not supported")
        elif DEVICE == "cuda":
            detection_model = pw_detection.HerdNet(device=DEVICE, dataset="ennedi")
    else:
        detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version=model)

    cliped_frames_path, count, meta_data, create_img_pathes = video_clip(im_file, num_segments_per_sec=video_sep)

    print('Video lenght is {} seconds. Separation between frames is {} seconds. Total frames: {}'.format(meta_data[36868] - meta_data[36867], video_sep, count))
    video_start = meta_data[36867]


    # Performing the detection on the single image
    if model == "HerdNet":
        result = detection_model.single_image_detection(img=im_file)
        #print(result)
    else:
        try:
            results = detection_model.batch_image_detection(cliped_frames_path, batch_size=count, det_conf_thres=threshold)
        except:
            results = detection_model.batch_image_detection(cliped_frames_path, batch_size=count)
            print("threshold set defalut value")
        #print(im_file + " has " + str(len(results)) + " frames")

    
        #result['img_id'] = result['img_id'].replace("\\","/")


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
            Time = video_start + (len(animal_ns) - 1) / video_sep
            print(f'{animal_ns[-1]} Animal detected in {result["img_id"]}. Time:{Time}')
            if not os.path.exists(new_file):
            # copy the video to the new file
                shutil.copy(im_file, new_file)

    result_first['animal_ns'] = animal_ns
    result_first['object'] = max(animal_ns)

    #delete directory of cliped_frames_path
    if dir_remove == 1:
        shutil.rmtree(cliped_frames_path)
    elif dir_remove == 2:
        for i in range(len(animal_ns)):
            if animal_ns[i] == 0:
                os.remove(create_img_pathes[i])
        print("image files(animal detected) are saved in " + cliped_frames_path)
    else:
        print("image files are saved in " + cliped_frames_path)

    try:
        result['eventStart']  = video_start
        result['eventEnd'] = meta_data[36868]
        result["Make"] = meta_data[271]
    except:
        result['eventStart'] = "None"
        result['eventEnd'] = "None"
        result["Make"] = None
    
    return result_first
