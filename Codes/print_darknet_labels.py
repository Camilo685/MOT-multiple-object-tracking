import darknet as dn
import numpy as np
import cv2

for x in range(1, 21):
	imagePath = ("/home/camilo685/Desktop/images_test/Ground_truth/Original_labels/image" + str(x) + ".png")
	
	r = dn.performDetect(imagePath, 0.25, "/home/camilo685/darknet/cfg/yolov3-tiny.cfg", "/home/camilo685/darknet/yolov3-tiny.weights", "/home/camilo685/darknet/cfg/coco.data", False, False, False)
	
	boxes=[]
	scores=[]
	classes=[]
	clean_b=[]
	
	img = cv2.imread(imagePath)
	
	with open("/home/camilo685/Desktop/images_test/Yolov3_tiny_detections_original/Yolov3_tiny_detection_original" + str(x), "w") as f:	
		for i in range(0,len(r)):
			if(r[i][1] > 0.35):
				if(r[i][0] == 'car'):
					boxes.append(np.asarray(r[i][2]))
					scores.append(r[i][1])
					classes.append(r[i][0])
					num_detections=len(r)
					x1, y1, x2, y2 = r[i][2]
					xl = int(x1 - x2/2)
					if xl < 0:
						xl = 0
					yb = int(y1 + y2/2)
					if yb > 375:
						yb = 375
					xr = int(x1 + x2/2)
					if xr > 1242:
						xr = 1242
					yt = int(y1 - y2/2)
					if yt < 0:
						yt = 0
					img = cv2.rectangle(img, (xl, yb), (xr, yt), (0, 0, 255), 2)
					row = np.array([xl, yb, xr, yt], dtype = float)
					f.write(np.array_str(row) + "\n")
		
	cv2.imshow('IOU', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(scores)
	
