#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:51:33 2017

@author: kyleguan
"""
import numpy as np
import cv2
import json
import perspective_change
import Distance_to_car
import os

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);



def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/(s_a + s_b -s_intsec)



def convert_to_pixel(box_yolo, img, crop_range):
    '''
    Helper function to convert (scaled) coordinates of a bounding box
    to pixel coordinates.

    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041,
    0.36866588651069609)

    crop_range: specifies the part of image to be cropped
    '''

    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape

    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)

    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))

    # Deal with corner cases
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0

    # Return the coordinates (in the unit of the pixels)

    box_pixel = np.array([left, top, width, height])
    return box_pixel

def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])

    return (left, top, right, bottom)

def draw_line_future(img,point1,point2,color):
    cv2.line(img,(int(point1[0]),int(point1[1])),(int(point2[0]),int(point2[1])), 255, 2)
    return img

def draw_line_traker(img,centroids,color):
    cen1=centroids[0]
    cen2=centroids[len(centroids)-1]
    print (cen1,cen2)
    cv2.line(img,(cen1[0],cen1[1]),(cen2[0],cen2[1]), color, 4, 8, 0)
    return img

def draw_poliLine_traker(img,centroids,color):
    points=np.array(centroids).reshape((-1,1,2)).astype(np.int32)
    #cv2.drawContours(img,[points],-1,color,3)
    cv2.polylines(img, [points], False, color, 1);
    return img
#def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
def draw_box_label(img, bbox_cv2, box_color, id ,box_colorP, distance, angle, vel, frame_weight, frame_height, show_distance = True, warning = True, show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    overlay=img.copy()
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)
    cv2.rectangle(overlay, (left, top), (right, bottom), box_colorP,-1,1) # representa la probabilidad de choque
    alpha=0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        #cv2.rectangle(img, (left-2, top-20), (right+2, top), box_color, -1, 1)
        cv2.rectangle(img, (left-2, top-20), (right+2, top), box_color, -1, 1)
        # Output the labels that show the x and y coordinates of the bounding box center.
        # text_x= 'x='+str((left+right)/2)
        # cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        # text_y= 'y='+str((top+bottom)/2)
        # cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
        #cv2.putText(img,str(round(velocity[-1],2)),(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
        cv2.putText(img,str(id),(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
        #cv2.putText(img,str(id)+str(street.id)+str(trk.direction),(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
        
    if show_distance:
		dist = "Distance (Meters) = " + str(distance)
		ang = "Angle (degrees) = " + str(angle)
		velo = "Velocity (k/h) = " +str(vel)
		cv2.putText(img, dist, (left,bottom-5), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.putText(img, ang, (left,bottom+15), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.putText(img, velo, (left,bottom+25), font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
		
		textsize = cv2.getTextSize("Stop", font, 5, 8)[0]
		textX = (frame_weight - textsize[0]) / 2
		textY = textsize[1] + 10
    
    if warning == True:
        if(distance <= 3 and (angle >= 50 and angle <= 130)):
        	cv2.putText(img, 'Stop', (textX, textY), font, 5, (255, 0, 0), 8, cv2.LINE_AA)
        	
    return img

def exist(Video_name, img):
	if os.path.exists('Camera_Calibration.txt'):
		with open('Camera_Calibration.txt') as json_file:  
			data = json.load(json_file)
			x = Video_name in data
			if(x == True):
				for item in data[Video_name]:
					M = np.asarray(item['Camera_matrix'])
					fl = item['Focal_length']
					return M, fl
			else:
				M = perspective_change.main(img)
				M = M.tolist()
				fl = Distance_to_car.main(img)
				data[Video_name] = []
				data[Video_name].append({
				'Camera_matrix': M,
				'Focal_length': fl
				})

				with open('Camera_Calibration.txt', 'w') as outfile:  
					json.dump(data, outfile)
				M = np.asarray(M)
				return M, fl
	else:
		M = perspective_change.main(img)
		M = M.tolist()
		fl = Distance_to_car.main(img)
		data = {}
		data[Video_name] = []
		data[Video_name].append({
		'Camera_matrix': M,
		'Focal_length': fl
		})

		with open('Camera_Calibration.txt', 'w') as outfile:  
			json.dump(data, outfile)
		M = np.asarray(M)
		return M, fl
