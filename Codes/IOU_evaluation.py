import glob, os
import numpy as np
import re
import cv2
import math
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment

result = ""
dum = np.array([1242, 375, 1242, 375], dtype = float)

def bb_intersection_over_union(a, b, c, d, e, f, g, h):
	##top left, bottom right, x1,y1,x2,y2
	xA = max(b, f)
	yA = max(a, e)
	xB = min(d, h)
	yB = min(c, g)
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (d - b + 1) * (c - a + 1)
	boxBArea = (h - f + 1) * (g - e + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

for x in range(1, 21):
	print("Image " + str(x))	
	center_truth = []
	truth = []
	img = cv2.imread(("Ground_truth/Original_labels/image" + str(x) + ".png"))
	det_with_truth = img.copy()
	
	with open(("Ground_truth/label" + str(x) + ".txt"), "r") as f:
		lines = f.readlines()
		for line in lines:
			result = re.findall(r"[-+]?\d*\.\d+|\d+", line)
			result = np.array(result, dtype = float)
			result = np.multiply(result, dum)
			xl = int(result[0] - result[2]/2)
			if xl < 0:
				xl = 0
			yb = int(result[1] + result[3]/2)
			if yb > 375:
				yb = 375
			xr = int(result[0] + result[2]/2)
			if xr > 1242:
				xr = 1242
			yt = int(result[1] - result[3]/2)
			if yt < 0:
				yt = 0
			center_truth.append([int(result[0]),int(result[1])])
			truth.append([int(yt), int(xl), int(yb), int(xr)])
			det_with_truth = cv2.rectangle(det_with_truth, (int(xl), int(yb)), (int(xr), int(yt)), (255, 0, 0), 2)
	
	with open(("Yolov3_tiny_detections_original/Yolov3_tiny_detection_original" + str(x)), "r") as m:		
	#with open(("Yolov3_detections_original/Yolov3_detection_original" + str(x)), "r") as m:		
	#with open(("Inception_v2_detections_original/inception_label_original" + str(x) + ".txt"), "r") as m:
	#with open(("Mobilenet_v2_detections_original/mobilenet_labels_original" + str(x) + ".txt"), "r") as m:
	#with open(("Resnet_detections_original/resnet_labels_original" + str(x) + ".txt"), "r") as m:
		detection = []
		IOU_scores = []
		lines = m.readlines()
		if len(center_truth) > len(lines):
			size = len(center_truth)
		else:
			size = len(lines)
		IOU_mat = np.zeros((size, size), dtype=np.float32)		
		count = 0
		for line in lines:
			result = re.findall(r"[-+]?\d*\.\d+|\d+", line)
			result = np.array(result, dtype = float)
			detection.append([int(result[3]), int(result[0]), int(result[1]), int(result[2])])
			p1, p2 = [int((result[0] + result[2])/2), (int(result[1] + result[3])/2)]
			det_with_truth = cv2.rectangle(det_with_truth, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 2)
			for s in range(0, len(center_truth)):
				x1, y1 = center_truth[s]
				euclidean_distance = math.sqrt((x1 - p1)**2 + (y1 - p2)**2 )
				IOU_mat[count, s] = euclidean_distance
			count = count + 1		
		index = linear_assignment(IOU_mat)		
		#print("IOU", IOU_mat)
		#print("Truth", truth)
		#print("Detection", detection)
		#print("Ass", col)		
		
		rel = 0
		for t in range(0, len(detection)):
			row, col = index[t]
			if col < len(truth):
				y1, x1, y2, x2 = truth[col]				
				yd1, xd1, yd2, xd2 = detection[t]				
				if(IOU_mat[t, col] < (y1 + y2)/4):
					if(IOU_mat[t, col] < (x1 + x2)/4):
						score = np.around(bb_intersection_over_union(y1, x1, y2, x2, yd1, xd1, yd2, xd2), decimals = 3)
						text = "IOU " + str(score)
						if score < 0.5:
							IOU = "IOU (Poor) = " + str(score)
						elif score < 0.9:
							IOU = "IOU (Good) = " + str(score)
						else:
							IOU = "IOU (Excellent) = " + str(score)
						cv2.putText(img, text, (int((x1 + x2) / 2), int(y1 - y1*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
						img = cv2.rectangle(img, (x1, y2), (x2, y1), (255, 0, 0), 2)
						img = cv2.rectangle(img, (xd1, yd2), (xd2, yd1), (0, 255, 0), 2)
						IOU_scores.append(score)
						rel = rel + 1
						print(IOU)
					else:
						img = cv2.rectangle(img, (xd1, yd2), (xd2, yd1), (0, 0, 255), 2)						
				else:
					img = cv2.rectangle(img, (xd1, yd2), (xd2, yd1), (0, 0, 255), 2)
					
		if (len(IOU_scores) != 0): 
			score_p = np.mean(IOU_scores)
		else:
			score_p = "No detection"
		
		print("The number of true detections is =", len(truth))
		print("The media for the image is =", score_p)
		print("The number of matches is =", len(IOU_scores))
		print("The number of misdetections is =", len(detection) - rel)
		print("The number of non detections is =", len(truth) - len(IOU_scores))
	
	
										
	cv2.imwrite(("Yolov3_tiny_detections_original/Labels+Detections_" + str(x) + ".png"), det_with_truth)
	cv2.imshow('Det_Tru', det_with_truth)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(("Yolov3_tiny_detections_original/Yolov3_tiny_IOU_" + str(x) + ".png"), img)
	cv2.imshow('IOU', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
