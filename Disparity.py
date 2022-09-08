import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


##### Use this if Left and Right are in the same image 
## Read the image
#img = cv2.imread(r'disp2.jpg')
#height = img.shape[0]
#width = img.shape[1]

## Cut the image in half
#width_cutoff = width // 2
#left = img[:, :width_cutoff]
#right = img[:, width_cutoff:]



#### Use this if Left and Right are in separate images

left = cv2.imread(r'p1_13_L_2034.jpg')
right = cv2.imread(r'p1_13_R_2034.jpg')


#Convert to Grayscale
left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
# Compute Disparity Map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left,right)
# Output picture
cv2.imwrite('output.jpg', disparity)

