#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:59:55 2021

@author: TEB
"""

from imageai import Detection
import cv2


######## YOU ONLY LOOK ONCE (YOLO) #######

# Note this is the easiest solution to the problem as I do not require to train a network as I can load preloaded weights.
# This is just to use as a benchmark to compare how my technique does against the state of the art solution.
# This open source YOLO model built can recognize cats, birds and 78 more object types.
# The database to train the model was created by Moses Olafenwa and the training was done by him as well
# This way I get around not having an NVIDIA GPU on my MAC Personal Computer (no access to CUDA for fast training) 


### Put your filepath to yolo.h5
MODELPATH = "/Users/TEB/IntuitiveSurgical/Interview/yolo.h5"

# YOLO CNN Detecting the
class YOLO:
    def __init__(self, path):
        # Setup the YOLO CNN with the imageAI library
        self.Yolo = Detection.ObjectDetection()
        # Load the model type
        self.Yolo.setModelTypeAsYOLOv3()
        # Set the path to the yolo.h5 file which has the pre-trained model weights
        self.Yolo.setModelPath(path)
        # Load the YOLO model's weights
        self.Yolo.loadModel()
    
    def pairYOLO2PCWebcam(self,
                          camera_frame_width = 1000,
                          camera_frame_height = 1000,
                          probability_threshold = 70,
                          show_probability = True,
                          show_name = True):
        
        # Activate the front camera
        # Note: 0 is the front camera & 1 is the back camera
        camera = cv2.VideoCapture(0)
        # Camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame_height)
        
        opticFlowActive = True
        
        while opticFlowActive:
            # read frames
            ret, img = camera.read()
            # Apply the YOLO mask to showcase the objects
            img, preds = self.Yolo.detectCustomObjectsFromImage(input_image=img,
                                                                custom_objects = None,
                                                                input_type = "array",
                                                                output_type = "array",
                                                                minimum_percentage_probability = probability_threshold,
                                                                display_percentage_probability = show_probability,
                                                                display_object_name = show_name)
            # Show predictions
            cv2.imshow("Video", img)
            
            
            # Use "q" or "ESCAPE" key to stop the optic flow.
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                opticFlowActive = False
                break
        
        # Close the camera
        camera.release()
        cv2.destroyAllWindows()
        print("WEBCAM OFF!")

if __name__ == '__main__':
    # Load the model
    yolo = YOLO(MODELPATH)
    # Activate the computer webcam and watch the results 
    yolo.pairYOLO2PCWebcam()