from imutils import paths
import numpy as np
import imutils
import cv2
import os
import json

ix, iy = -1, -1
img = None
box = np.zeros((2, 2), dtype = "float32")
click_count = 0
KNOWN_HEIGTH = 0
KNOWN_DISTANCE = 0

def main(frame):
	global img, box, KNOWN_HEIGTH, KNOWN_DISTANCE
	img = frame
	cv2.namedWindow('Focal_length_calculation')
	cv2.setMouseCallback('Focal_length_calculation', draw_circle)
	
	while(click_count < 2):
		cv2.imshow('Focal_length_calculation', img)
		k = cv2.waitKey(20) & 0xFF
		if k == 27:
		    break

	KNOWN_DISTANCE = input("Known distance")
	tipe = input("Tipe")
	if(tipe == "car"):
		KNOWN_HEIGTH = 1500
	else:
		KNOWN_HEIGTH = 2800	
	
	img = cv2.rectangle(img,(box[0, 0], box[0, 1]),(box[1, 0], box[1, 1]),(0, 255, 0), 3)
	if (box[0, 1] > box[1, 1]):
		object_pix_heigth = box[0, 1] - box[1, 1]		
	else:
		object_pix_heigth = box[1, 1] - box[0, 1]
	focalLength = (object_pix_heigth * KNOWN_DISTANCE) / KNOWN_HEIGTH
#	distance = (KNOWN_HEIGTH * focalLength) / object_pix_heigth
#	print("Focal = ", focalLength, "Distance = ", distance)
	cv2.imshow('Rectangle_image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return focalLength

def draw_circle(event, x, y, flags, param):
	global ix, iy, img, box, click_count
	if event == cv2.EVENT_LBUTTONDBLCLK and click_count < 2:
		cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
		ix, iy = x, y
		box[click_count, :] = x, y
		click_count = click_count + 1

if __name__ == "__main__":
	frame = cv2.imread("/home/camilo685/Desktop/Tesis/Code/image_calibration.jpg")
	main(frame)
