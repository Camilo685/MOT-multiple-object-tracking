import cv2
import numpy as np
import imutils

ix, iy = -1, -1
img = None
box = np.zeros((4, 2), dtype = "float32")
click_count = 0

def main(frame):
	global img, box
	img = frame
	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	if (img.shape[1]>1300):
		img = imutils.resize(img, width = 1200)
		
	cv2.namedWindow('Camera_Calibration', cv2.WINDOW_NORMAL)
	cv2.setMouseCallback('Camera_Calibration', draw_circle)

	while(click_count < 4):
		cv2.imshow('Camera_Calibration', img)
		k = cv2.waitKey(20) & 0xFF
		if k == 27:
		    break
#	box = order_points(box)
	persp, M = four_point_transform(img, box)
	
	cv2.imshow('Final image', persp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return M

def order_points(box):
	rect = np.zeros((4, 2), dtype = "float32")
	s = box.sum(axis = 1)
	rect[0] = box[np.argmin(s)]
	rect[3] = box[np.argmax(s)]
	d = np.diff(box, axis = 1)
	rect[1] = box[np.argmin(d)]
	rect[2] = box[np.argmax(d)]
	return rect

def four_point_transform(img, box):
	(tl, tr, br, bl) = box
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	M = cv2.getPerspectiveTransform(box, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	return warped, M 

def draw_circle(event, x, y, flags, param):
	global ix, iy, img, box, click_count
	if event == cv2.EVENT_LBUTTONDBLCLK and click_count < 4:
		cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
		ix, iy = x, y
		box[click_count, :] = x, y
		click_count = click_count + 1

if __name__ == "__main__":
	frame = cv2.imread("/home/camilo685/Desktop/final_project/dataset/train_real/images/Imagen34.jpg")
	main(frame)
