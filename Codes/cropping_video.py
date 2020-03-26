import cv2
import imutils

def pipeline(img):
	if (img.shape[0]>600):
		img = imutils.resize(img, height = 600)
	#cv2.imshow("original", img)
	#cv2.waitKey(0)
	cropped = img[220:, :]
	return cropped

if __name__ == "__main__":
	cap = cv2.VideoCapture("/home/camilo685/Desktop/example_code/test.mp4")
	filename = '/home/camilo685/Desktop/example_code/test_salida.avi'
	codec = cv2.VideoWriter_fourcc('M','J','P','G')
	fps = 12
	Output = cv2.VideoWriter(filename, codec, fps, (1066,380))
			
	while(True):
		ret, frame = cap.read()
		if ret == True:
			img = pipeline(frame)
			Output.write(img)
			cv2.imshow('frame',img)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
			
	Output.release()
	cap.release()
	cv2.destroyAllWindows()
