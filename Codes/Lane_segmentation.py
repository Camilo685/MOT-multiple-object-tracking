import cv2
import sys, getopt
import numpy as np
import math
from matplotlib import pyplot as plt

def processing(img):
	imshape = img.shape
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	
	
    
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img1 = smoothing(gray, 5)
	img1 = canny(img1, 50, 150)

	plt.subplot(121),plt.imshow(img)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img1, cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

### Video's vertices
	lower_left = [0, imshape[0]]
	lower_right = [imshape[1], imshape[0]]
	top_left = [0, imshape[0]/1.5]
	top_right = [imshape[1], imshape[0]/1.5]

	vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype = np.int32)]

	img1  = ROI(img1, vertices)

	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img1, cmap = 'gray')
	plt.title('ROI edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()

	lines = cv2.HoughLinesP(img1, rho = 4, theta = (np.pi/180), threshold = 100, lines = np.array([]), minLineLength = 40, maxLineGap = 150)

	img2 = draw_lines(img, lines)
	
	plt.subplot(121),plt.imshow(img)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img2,cmap = 'gray')
	plt.title('Original Image with lines'), plt.xticks([]), plt.yticks([])
	plt.show()

def smoothing(img, ksize):
	return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny(img, low_trhld, high_trhld):
	return cv2.Canny(img, low_trhld, high_trhld)

def ROI(img, vertices):
	mask = np.zeros_like(img)
	color_mask = 255
	cv2.fillPoly(mask, vertices, color_mask)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color = [0, 0, 255], thickness = 5):
	if lines is None:
		return img

	line_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)

	left_line_x = []
	left_line_y = []
	right_line_x = []
	right_line_y = []

	for line in lines:
		for x1, y1, x2, y2 in line:
			if(x2-x1 == 0):
				print('dividing by zero')
				return img

			slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
			if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
				continue
			if slope <= 0: # <-- If the slope is negative, left group.
				left_line_x.extend([x1, x2])
				left_line_y.extend([y1, y2])
			else: # <-- Otherwise, right group.
				right_line_x.extend([x1, x2])
				right_line_y.extend([y1, y2])

	min_y = int(img.shape[0] * (3.5 / 5)) # <-- Just below the horizon
	max_y = int(img.shape[0]) # <-- The bottom of the image

	if ((len(left_line_x) is 0) or (len(left_line_y) is 0) or (len(right_line_x) is 0) or (len(right_line_y) is 0)):
		print("No lane detected")
		return img

	poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

	left_x_start = int(poly_left(max_y))
	left_x_end = int(poly_left(min_y))

	poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

	right_x_start = int(poly_right(max_y))
	right_x_end = int(poly_right(min_y))
	
	new_lines = [[[left_x_start, max_y, left_x_end, min_y], [right_x_start, max_y, right_x_end, min_y]]]

	for line in new_lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

	img = cv2.addWeighted(img, 1, line_img, 1.0, 0.0)	

	return img
	
if __name__ == "__main__":
	processing(cv2.imread("/home/camilo685/Desktop/Tesis/Code/image.jpg"))
