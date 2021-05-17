#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:14:27 2021

@author: TEB
"""



### SmartCamera Software designed by Teddy Edmond Benkohen


# Import openCV
import cv2
# Import numpy
import numpy as np
# import the load model function from Keras
from keras.models import load_model
# import image tools from Keras
from keras.preprocessing import image


# ### UN/COMMENT TO TEST THE BirdsVsCats1 model.
# ### Put your filepath to BirdsVsCats1 model.
# MODELPATH = "/Users/TEB/IntuitiveSurgical/Interview/BirdsVsCats1.h5"

### UN/COMMENT TO TEST THE BirdsVsCats2 model.
### Put your filepath to BirdsVsCats2 model.
MODELPATH = "/Users/TEB/IntuitiveSurgical/Interview/BirdsVsCats2.h5"

class SmartCamera:
    def __init__(self, path):
        self.BirdsVsCats = load_model(path)
    
    def outlinePictures(self,
                        gray,
                        morphology = 7,
                        tolerance = 0.025,
                        bilateral_filter_parameters = [1, 10, 120],
                        gaussian_blur_filter_parameters = [5, 5, 1],
                        canny_filter_parameters = [30, 200]):
        
        # array containing the relevant contours
        rectangle_contours = []
        # Bilateral filter Accentuates the edges
        gray = cv2.bilateralFilter(gray,
                                   bilateral_filter_parameters[0],
                                   bilateral_filter_parameters[1],
                                   bilateral_filter_parameters[2])
        # Blur filter.
        gray = cv2.GaussianBlur(gray,
                                (gaussian_blur_filter_parameters[0],
                                 gaussian_blur_filter_parameters[1]),
                                gaussian_blur_filter_parameters[2])
        # Find Canny edges
        gray = cv2.Canny(gray,
                         canny_filter_parameters[0],
                         canny_filter_parameters[1])
        # get the rectangle kernel
        rectangle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                     (morphology, morphology))
        # check if the rectangle is closed
        closed = cv2.morphologyEx(gray,
                                  cv2.MORPH_CLOSE,
                                  rectangle_kernel)
        # Finding Contours
        contours, hierarchy = cv2.findContours(closed,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        
        
        # Select only rectangles
        for i in contours:
            # The epsilon is the error tolerated in the recognition
            epsilon = tolerance * cv2.arcLength(i, True)
            # approximate the number of sides in the closed figure
            approx = cv2.approxPolyDP(i, epsilon, True)
            # if four sides are found then it is a rectangle
            if len(approx) == 4:
               rectangle_contours.append(approx)
        
        # return rectangle contours 
        return rectangle_contours
    
    def predict(self, img):
        # resize the image to fit the CNN
        img = cv2.resize(img, (64, 64))
        # Image Preprocessing
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        # predict if there is a bird or a cat.
        result = self.BirdsVsCats.predict(img)
        return result
    
    def print_one_node_network_output(self, frame, result, x, y, w, h):
        # Print the output of the one node output network CNN
        
        if result[0][0] == 0:
            # Show the CNN recognizes the bird
            # Draw a green rectangle around the bird
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            # Write Bird in green over the top left corner of the rectangle
            cv2.putText(frame, "Bird", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Show the CNN recognizes the cat
            # Draw a red rectangle around the cat
            cv2.rectangle(frame, (x, y), (x+ w, y+h), (0, 0, 255), 2)
            # Write Cat in red over the top left corner of the rectangle
            cv2.putText(frame, "Cat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def print_two_node_network_output(self, frame, result, x, y, w, h, threshold = 0.95):
        # find the class label index with the largest corresponding probability
        result1 = result[0][0]
        result2 = result[0][1]
        
        # Print the output of the one node output network CNN
        if result1 > result2 and result1 > threshold:
            # Show the CNN recognizes the bird
            # Draw a green rectangle around the bird
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            # Write Bird in green over the top left corner of the rectangle
            cv2.putText(frame, "Bird", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif result1 < result2 and result2 > threshold:
            # Show the CNN recognizes the cat
            # Draw a red rectangle around the cat
            cv2.rectangle(frame, (x, y), (x+ w, y+h), (0, 0, 255), 2)
            # Write Cat in red over the top left corner of the rectangle
            cv2.putText(frame, "Cat", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            pass
            
    
    def detect(self, ret, frame):
        # turn the frame gray for faster edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get the outline of the picture
        contours = self.outlinePictures(gray)
        
        # Here for loop iterate over the contours
        for contour in contours:
            ## Note:(x, y) are the coordinates of the bottom left corner, w is the width and h the height
            # Get the coordinates
            (x, y, w, h) = cv2.boundingRect(contour)
            # Get the region of interest by cropping the contour
            roi = frame[y:y+h, x:x+w]
            # Apply the CNN
            result = self.predict(roi)
            
            # ## UN/COMMENT TO SEE OR NOT SEE THE OUTPUT
            # # Print the output of the one node output network CNN FOR BirdsVsCats1.h5
            # self.print_one_node_network_output(frame, result, x, y, w, h)
            
            ## UN/COMMENT TO SEE OR NOT SEE THE OUTPUT
            # Print the output of the two node output network CNN FOR BirdsVsCats2.h5
            self.print_two_node_network_output(frame, result, x, y, w, h)
            
        # Play the optic frow of the Smart Camera on your laptop :D
        cv2.imshow('Video', frame)
    
    def pairWebcam2Detector(self, camera_frame_width = 1000, camera_frame_height = 1000):
        camera = cv2.VideoCapture(0)
        # Camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame_height)
        
        opticFlowActive = True
        
        while opticFlowActive:
            # read frames
            ret, frame = camera.read()
            # Apply the Custom CNN Mask to identify birds vs cats in the frame
            self.detect(ret, frame)
            
            # Use "q" or "ESCAPE" key to stop the optic flow.
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                opticFlowActive = False
                break
        
        # Close the camera
        camera.release()
        cv2.destroyAllWindows()
        print("WEBCAM OFF!")
        return


if __name__ == '__main__':
    smartCamera = SmartCamera(MODELPATH)
    smartCamera.pairWebcam2Detector()






        