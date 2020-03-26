#!/usr/bin/env python2
'''
Script to test traffic light localization and detection
'''

import numpy as np
import cv2
from PIL import Image
import sys, os
from matplotlib import pyplot as plt
import time
#from glob import glob
#cwd = os.path.dirname(os.path.realpath(__file__))
#cwd2= os.path.join(os.getcwd(),'')
#sys.path.append(cwd2)
import darknet as dn



class CarDetector(object):
    def __init__(self):

        self.car_boxes = []
        #os.chdir(cwd2)
        #YoloV3
        #self.net = dn.load_net(b"/home/camilo685/darknet/cfg/yolov3.cfg", b"/home/camilo685/darknet/yolov3.weights", 0)
        #YoloV3-Tiny
        self.net = dn.load_net(b"/home/camilo685/darknet/cfg/yolov3-tiny.cfg", b"/home/camilo685/darknet/yolov3-tiny.weights", 0)
        #YoloV2-Tiny
        #self.net = dn.load_net("cfg/yolov2-tiny.cfg", "yolov2-tiny-voc.weights", 0)
        self.meta = dn.load_meta(b"/home/camilo685/darknet/cfg/coco.data")

        print (self.meta)
        # file=open("logs.txt","a")
        # file.write("-------------------------------------\n")
        # file.write(self.meta)
        # file.write("idx 1: %f\n" %maxiooa_1[1])
        # file.write("-------------------------------------\n")
        # file.close()



        self.boxes =[]
        self.scores=[]
        self.classes=[]
        self.num_detections=[]


    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
         
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
        #para trabajar con el detector de tensorflow y el traker filtro de kalman
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def box_normal_to_pixel2(self, box):
        # para trabajar con el detector yolo y traker filtro de kalman
        box_pixel=[int(box[1]-(box[3]/2)),int(box[0]-(box[2]/2)),int(box[1]+(box[3]/2)),int(box[0]+(box[2]/2))]
        return np.array(box_pixel)

    def box_normal_to_pixel3(self, box):
        # para trabajar con el multitraker de opencv usando detector de yolo
        box_pixel=[int(box[0]-(box[2]/2)),int(box[1]-(box[3]/2)),box[2],box[3]]
        return np.array(box_pixel)


    def get_localizationYolo(self, image,file, visual=False):

        boxes=[]
        scores=[]
        classes=[]

        r = dn.detect(self.net, self.meta,file)
        for i in range(0,len(r)):
        	boxes.append(np.asarray(r[i][2]))
        	scores.append(r[i][1])
        	classes.append(r[i][0])
        num_detections=len(r)
        
        boxes=np.squeeze(boxes)
        classes =np.squeeze(classes)
        scores = np.squeeze(scores)

        cls = classes.tolist()
        #Ver detecciones
        print("Detecciones ", cls)
        # The ID for car is 3
        if isinstance(cls, list):
        	idx_vec = [i for i, v in enumerate(cls) if (v == "car" or v == "bus" or v == "truck") and scores[i] > 0.25]
			
        	if len(idx_vec) ==0:
        		print('no detection!')
        	else:
        		tmp_car_boxes=[]
            	for idx in idx_vec:
                	dim = image.shape[0:2]
                	box = self.box_normal_to_pixel2(boxes[idx])
                	cls = classes
                	box_h = box[2] - box[0]
                	box_w = box[3] - box[1]
                	
                	ratio = box_h/(box_w + 0.01)
                	#tmp_car_boxes.append(box)
                	if ((ratio < 0.95) and (box_h>15) and (box_w>15)):
                	#if ((ratio < 0.95) and (box_h>20) and (box_w>20)):
                		tmp_car_boxes.append(box)
                		print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                		
                	else:
                		print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                		
                	self.car_boxes = tmp_car_boxes        
        else:
        	if (cls != 'car'):
        		print('no detection!')
        		
        	else:
        		if scores > 0.25:
		    		tmp_car_boxes=[]
		    		dim = image.shape[0:2]
		    		box = self.box_normal_to_pixel2(boxes)
		    		cls = classes
		    		box_h = box[2] - box[0]
		    		box_w = box[3] - box[1]
		    		
		    		ratio = box_h/(box_w + 0.01)
		    		#tmp_car_boxes.append(box)
		    		if ((ratio < 0.95) and (box_h>15) and (box_w>15)):
		    		#if ((ratio < 0.95) and (box_h>20) and (box_w>20)):
		    			tmp_car_boxes.append(box)
		    			print(box, ', confidence: ', scores, 'ratio:', ratio)
		    			
		    		else:
		    			print('wrong ratio or wrong size, ', box, ', confidence: ', scores, 'ratio:', ratio)
		    			
		    		self.car_boxes = tmp_car_boxes
        return self.car_boxes, cls
