#!/usr/bin/env python2
from __future__ import division
import darknet
import detector
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import copy
import helpers
import tracker
import math
from collections import deque
import os
import imutils


tracker_list = []
track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
frame_count = 0
min_hits = 1
max_age = 5
focal_length = 0
car_height = 1500
truck_height = 2800
distance = 0
object_height = 0

#Video_Path = "/home/camilo685/Desktop/test.mp4"
Video_Path = "/home/camilo685/Desktop/Tesis/Code/test4.mp4"
#Video_Path = "/home/camilo685/calib_salida.mp4"
#Video_name = os.path.basename(Video_Path)
Video_name = "calib_salida.mp4"
Camera_matrix = 0


frameRate = 25

def box_iou2(a, b):
 
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    return float(s_intsec)/(s_a + s_b -s_intsec)

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
      
        for d,det in enumerate(detections):
            IOU_mat[t,d] = box_iou2(trk,det)

    # Hungarian algorithm
    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)
            
    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def calculate_centroid(xx,tmp_trk):
    cx=xx[1]+((xx[3]-xx[1])/2)
    cy=xx[0]+((xx[2]-xx[0])/2)
    centroid=np.array([cx,cy])

    if len(tmp_trk.centroids)>10:
        centroid_old=tmp_trk.centroids[len(tmp_trk.centroids)-9]
    else:
        centroid_old=tmp_trk.centroids[len(tmp_trk.centroids)-1]

    distancePix=np.sqrt(sum((centroid-centroid_old)**2))

    success=0
    if distancePix>0.9:
        tmp_trk.centroids.append(centroid)
        tmp_trk.frameNum.append(frame_count)
        success=1
#    print("Centroids", centroid)
#    print("New line ______________")

    return tmp_trk,success,distancePix

def pipeline(img):
	global frame_count
	global tracker_list
	global max_age
	global min_hits
	global track_id_list

#	global streetList
	global pathVideo
	global Video_name
	global Camera_matrix
	global focal_length
	global car_height
	global distance
	global object_height
	
	if (img.shape[1]>1300):
		img = imutils.resize(img, width = 1200)

	frame_count+=1

	image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	
	if(frame_count == 1):
		Camera_matrix, focal_length = helpers.exist(Video_name, img)
#		cv2.imwrite("image0.jpg",image)
#		print (Camera_matrix, focal_length)	

	cv2.imwrite("image.jpg",image)
	z_box, temp = det.get_localizationYolo(img, b'image.jpg') # measurement
	##z_box = coordinadas de las detecciones [a, b, c, d]
	##temp = clase de las detecciones ['det1', 'det2']
   
	img_dim = (img.shape[1], img.shape[0])
   
	x_box =[]
	##x_box = coordenadas de los tracker
	if len(tracker_list) > 0:
		for trk in tracker_list:
			x_box.append(trk.box)

	matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)
	
	# matched detections
	if matched.size >0:
		for trk_idx, det_idx in matched:
			z = z_box[det_idx]
			z = np.expand_dims(z, axis=0).T
			tmp_trk= tracker_list[trk_idx]
			tmp_trk.kalman_filter(z)
			xx = tmp_trk.x_state.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]

			x_box[trk_idx] = xx
			tmp_trk.box =xx
			tmp_trk.hits += 1

    # unmatched detections
	if len(unmatched_dets)>0:
		if (temp.size)>1:
			for idx in unmatched_dets:
				z = z_box[idx]
				z = np.expand_dims(z, axis=0).T
				tmp_trk = tracker.Tracker() # Create a new tracker
				x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
				tmp_trk.cls = temp[idx]
				tmp_trk.x_state = x
				tmp_trk.predict_only()
				xx = tmp_trk.x_state
				xx = xx.T[0].tolist()
				xx =[xx[0], xx[2], xx[4], xx[6]]

		        #-------fist time-------------------
				cx=xx[1]+((xx[3]-xx[1])/2)
				cy=xx[0]+((xx[2]-xx[0])/2)
				centroid=np.array([cx,cy])
				tmp_trk.centroids.append(centroid)
				tmp_trk.frameNum.append(frame_count)
		        #------------------------------------------------

				tmp_trk.box = xx
				tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
				tracker_list.append(tmp_trk)
				x_box.append(xx)
		else:
			z = z_box
			z = np.expand_dims(z, axis=0).T
			tmp_trk = tracker.Tracker() # Create a new tracker
			x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
			tmp_trk.cls = temp
			tmp_trk.x_state = x
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]

	        #-------fist time-------------------
			cx=xx[1]+((xx[3]-xx[1])/2)
			cy=xx[0]+((xx[2]-xx[0])/2)
			centroid=np.array([cx,cy])
			tmp_trk.centroids.append(centroid)
			tmp_trk.frameNum.append(frame_count)
		        #------------------------------------------------

			tmp_trk.box = xx
			tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
			tracker_list.append(tmp_trk)
			x_box.append(xx)

    # unmatched tracks
	if len(unmatched_trks)>0:
		for trk_idx in unmatched_trks:
			tmp_trk = tracker_list[trk_idx]
			tmp_trk.no_losses += 1
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]

			tmp_trk.box =xx
            #tracker_list_10f[trk_idx]=copy.copy(tmp_trk)
			x_box[trk_idx] = xx


    # The list of tracks to be annotated
	good_tracker_list =[]
	imgray=img.copy()
	for trk_idx,trk in enumerate(tracker_list):
    #for trk in tracker_list:
		if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
			#--------------collision features-----------------
			xx=trk.box
			box = np.array(((xx[1] + xx[3]) / 2, xx[2]), dtype = 'float32').reshape(1, 1, -1)
#			box = np.array([[xx[1], xx[0]], [xx[3], xx[0]], [xx[1], xx[2]], [xx[3], xx[2]]], dtype='float32')
			print("Original points:")
			print(box)
#			box = np.array([box])
			pointsOut = cv2.perspectiveTransform(box, Camera_matrix).reshape(-1, 1)
			print("Perspective points:")
			print(pointsOut)
#			dis_x, dis_y = abs(pointsOut/)
			print("Distance(M)= ", distance)
			trk,success,distancePix=calculate_centroid(xx,trk)
			
			if(trk.cls == "car"):
				object_height = car_height
			elif(trk.cls == "truck"):
				object_height = truck_height
							
#			print(object_height, " tipe ", trk.cls)
             #--------------------------------------------------
			x_cv2 =trk.box
			object_pix_heigth = (xx[2] - xx[0])
			if object_pix_heigth == 0:
				object_pix_heigth = 1
			temp_dist = trk.distance
			distance = round((((object_height * focal_length) / object_pix_heigth) / 1000), 3)
			diff = distance - temp_dist
			trk.distance = distance
#			print("old= ", temp_dist, "new= ", distance)
			vel = round((diff / (1/25))*3.6, 3)
#			print("Vel= ", vel)
			
			dx = (xx[1]+((xx[3]-xx[1])/2)) - img_dim[0]/2
			dy = xx[2]
			angle = math.atan2(dy, dx)
			angle = round(math.degrees(angle), 3)
			show_dis = True
			warning = False
			show_lab = True
			traffic_light = True
						
			img= helpers.draw_box_label(img, x_cv2, trk.box_color, trk.id, trk.colorProb, distance, angle, vel, img.shape[1], img.shape[0], show_dis, warning, show_lab)
			
			if show_dis == True:
				img= helpers.draw_poliLine_traker(img,trk.centroids,trk.box_color)
				img = cv2.line(img,(int(img_dim[0]/2),img_dim[1]),(int(img_dim[0]/2 + img_dim[0]/4), img_dim[1]),(255,0,0),2)
				img = cv2.line(img,(int(img_dim[0]/2),img_dim[1]),(int((xx[1]+((xx[3]-xx[1])/2))),xx[2]),(255,0,0),2)
				img = cv2.ellipse(img,(int(img_dim[0]/2),img_dim[1]),(10,10),0,0,-angle,255,-1)
			
			if traffic_light == True:			
				if(distance <= 3 and (angle >= 50 and angle <= 130)):
					img = cv2.circle(img, (int(img.shape[1]/2) - 100, 60), 50, (255, 0, 0), -1)
					img = cv2.circle(img, (int(img.shape[1]/2) + 100, 60), 50, (96, 96, 96), -1)
				else:
					img = cv2.circle(img, (int(img.shape[1]/2) - 100, 60), 50, (96, 96, 96), -1)
					img = cv2.circle(img, (int(img.shape[1]/2) + 100, 60), 50, (0, 255, 0), -1)

    # Book keeping
	deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

	for trk in deleted_tracks:
		track_id_list.append(trk.id)

	tracker_list = [x for x in tracker_list if x.no_losses <= max_age]
	return img

if __name__ == "__main__":
	det = detector.CarDetector()
	output = 'salidat.mp4'
	clip1 = VideoFileClip(Video_Path)
	clip1 = clip1.set_fps(frameRate)
	clip = clip1.fl_image(pipeline)
	clip.write_videofile(output, audio = False, fps = frameRate)
	
