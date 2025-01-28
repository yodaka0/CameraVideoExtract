
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

    parser = createParser(im_file)
    if not parser:
        print("Unable to parse file")
        exifdata = None
    
    else:
        with parser:
            metadata = extractMetadata(parser)
            if not metadata:
                print("Unable to extract metadata")
            else:
                exifdata = metadata.exportDictionary()

    
    return cliped_frames_path, count, exifdata


def pw_detect(im_file, new_file, threshold=None, pre_detects=None, diff_reasoning=False, verbose=False, model=None, dir_remove=True):

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

    cliped_frames_path, count, exif_data = video_clip(im_file)


    #new_file_path = os.path.dirname(new_file)

    # Performing the detection on the single image
    #result = detection_model.single_image_detection(transform(img), img.shape, im_file, conf_thres=threshold)
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
        print(im_file + " has " + str(len(results)) + " frames")

    
        #result['img_id'] = result['img_id'].replace("\\","/")

     """   if diff_reasoning and pre_detects is not None:
            # extract values of bounding boxes from the result dictionary
            bounding_boxes = result['detections'].xyxy
            prev_bounding_boxes = pre_detects.xyxy
            # if bounding boxes heardly change, then animal change into blank
            if len(bounding_boxes) == len(prev_bounding_boxes):
                for i in range(len(bounding_boxes)):
                    if (abs(bounding_boxes[i][0] - prev_bounding_boxes[i][0]) < 5 and
                        abs(bounding_boxes[i][1] - prev_bounding_boxes[i][1]) < 5 and
                        abs(bounding_boxes[i][2] - prev_bounding_boxes[i][2]) < 5 and
                        abs(bounding_boxes[i][3] - prev_bounding_boxes[i][3]) < 5):
                        print(f"bounding boxes:{bounding_boxes[i]}")
                        print(f"previous bounding boxes:{prev_bounding_boxes[i]}")
                        print("bounding box not move, change to blank")
                        #transform animal in the labels to 'blank'
                        result['labels'][i] = "blank"""

    animal_ns = []
    first = True
    for result in results:
        animal_ns.append(sum('animal' in item for item in result['labels']))
        #save first result
        if first:
            first = False
            result_first = result
        #print(animal_ns[-1])
        if animal_ns[-1] > 0 and not os.path.exists(new_file):
            #print('Animal detected')
            # copy the video to the new file
            shutil.copy(im_file, new_file)

    result_first['animal_ns'] = animal_ns
    result_first['object'] = max(animal_ns)

    #delete directory of cliped_frames_path
    if dir_remove:
        shutil.rmtree(cliped_frames_path)

    try:
        result['eventStart']  = exif_data[36867]
        result['eventEnd'] = exif_data[36867]
        result["Make"] = exif_data[271]
    except:
        result['eventStart'] = "None"
        result['eventEnd'] = "None"
        result["Make"] = None
    
    return result_first


    """Saving the detection results 
    animal_n = sum('animal' in item for item in result['labels'])
    print(f'{im_file} has {animal_n} animals')
    result['object'] = animal_n



    if model == "HerdNet":
        pw_utils.save_detection_images_dots(result, new_file_path, overwrite=False)
    elif animal_n > 0:
        if verbose:
            print(f"Saving detection images to {new_file_path}")
            print(result)
        pw_utils.save_detection_images(result, new_file_path, overwrite=False)
        if classify:
            try:
                animalclass = Classifier(model_dir=os.getcwd())
                sp, conf = animalclass.run_prediction(result, img)
                result['scientific_name'] = sp
                result['sp_confidence'] = conf
            except Exception as e:
                print(f"Error in classification: {e}")
                raise

    return result"""
