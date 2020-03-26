import urllib
import cv2
import numpy as np
import time

url = 'http://192.168.1.52:8080/shot.jpg'

#while True:
#	imgweb = urllib.urlopen(url)
#	imgnp = np.array(bytearray(imgweb.read()),dtype = np.uint8)
#	img = cv2.imdecode(imgnp, -1)
#	cv2.imshow('test', img)
#	cv2.waitKey(10)
	
cap = cv2.VideoCapture('http://192.168.1.52:8080/video')
filename = '/home/camilo685/Desktop/Tesis/Code/salida_test.avi'
codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
resolution = (800, 600)

num_frames = 120;    
print "Capturing {0} frames".format(num_frames)

# Start time
start = time.time()
     
# Grab a few frames
for i in xrange(0, num_frames) :
    ret, frame = cap.read()
# End time
end = time.time()

# Time elapsed
seconds = end - start
print "Time taken : {0} seconds".format(seconds)
 
# Calculate frames per second
fps  = num_frames / seconds;
print "Estimated frames per second : {0}".format(fps);

Output = cv2.VideoWriter(filename, codec, fps, resolution)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	Output.write(frame)

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
	    
Output.release()
cap.release()
cv2.destroyAllWindows()
