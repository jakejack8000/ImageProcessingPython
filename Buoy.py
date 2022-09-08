import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep
# Definitions for annotations
font = cv2.FONT_HERSHEY_SIMPLEX    
fontScale = 0.5
   
red = (0, 0, 255)
blue =   (255, 0, 0)
green = (0,255,0)
white = (255,255,255)
yellow = (0,255,255)
thickness = 1
# Read Video
cap = cv2.VideoCapture(r'Bouys_1.mp4')
#cap = cv2.VideoCapture(r'Buoys_2.mp4')
#cap = cv2.VideoCapture(r'Red_Buoy.mp4')

###### Get video width and height and define writer codec
frame_height = int(cap.get(4))
frame_width = int(cap.get(3))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outboys2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

### define kernels for morphological opertions

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
bigkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
medkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
## Repeated Processing For each video frame
while True:
    # read the frame
    ret, img = cap.read()
    # create a copy for the result image output
    result = img.copy()
    # convert to LAB colorspace 
    imgy = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    imgy = cv2.cvtColor(imgy, cv2.COLOR_BGR2Lab)
    # Keep only yellow pixels , turn them to white on a black background
    imgy = cv2.inRange(imgy, np.array([150, 130, 150]), np.array([255, 170, 255]))
    # Morphological operations and gaussian filter to remove noise
    imgy = cv2.erode(imgy,kernel,iterations = 1)
    imgy = cv2.morphologyEx(imgy, cv2.MORPH_CLOSE, bigkernel)
    imgy = cv2.GaussianBlur(imgy, (7, 7), 2, 2)
    # An image called 'imgy' now has the locations of yellow pixels in original image as white in this image

    
    #Repeat the previous steps keeping green pixels instead
    
    imgg = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2Lab)
    imgg = cv2.inRange(imgg, np.array([130, 0, 145]), np.array([255, 75, 255]))
    
    imgg = cv2.erode(imgg,kernel,iterations = 1)
    imgg = cv2.morphologyEx(imgg, cv2.MORPH_CLOSE, bigkernel)

    imgg = cv2.GaussianBlur(imgg, (7, 7), 2, 2)
    
    # An image called 'imgg' now has the locations of green pixels in original image as white in this image
    #Repeat the previous steps keeping red pixels instead
    
    imgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2Lab)
    imgr = cv2.inRange(imgr, np.array([50, 150, 0]), np.array([100, 255, 255]))
    imgr = cv2.erode(imgr,kernel,iterations = 1)
    imgr = cv2.morphologyEx(imgr, cv2.MORPH_CLOSE, bigkernel)
    imgr = cv2.GaussianBlur(imgr, (7, 7), 2, 2)
    
    # An image called 'imgr' now has the locations of red pixels in original image as white in this image

    
    ## Perform connected components analysis to extract centroids of buoys
    y_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgy)
    for i in range(1, y_components):
        x = int(centroids[i,0]) 
        y = int(centroids[i,1]) + 20
        cv2.putText(result,'Yellow Buoy', (x,y), font, fontScale, yellow, thickness, cv2.LINE_AA )
        
        ## Creates a mask from 'imgy'
        mask = cv2.dilate(imgy,medkernel,iterations=1)
        #Passes the mask on the original frame to 'crop' the buoys
        masked = cv2.bitwise_and(img, img, mask=mask)
        # Convert to gray and threshold
        imgray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # find the minimum area circle around contours to circle the buoy
        contours, hier = cv2.findContours(thresh, 1, 2)
        radius_r = []
        if contours:
            for c in contours:
                area = int(cv2.contourArea(c))
                if area >> 5 :
                    (x,y),radius = cv2.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(result,center,radius,yellow,2)
                
                
                
                
    ## repeat the previous steps for green            
    g_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgg)
    for i in range(1, g_components):
        x = int(centroids[i,0]) 
        y = int(centroids[i,1]) + 20
        cv2.putText(result,'Green Buoy', (x,y), font, fontScale, green, thickness, cv2.LINE_AA )
        
        
        mask = cv2.dilate(imgg,medkernel,iterations=1)

        masked = cv2.bitwise_and(img, img, mask=mask)

        imgray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        contours, hier = cv2.findContours(thresh, 1, 2)
        radius_r = []
        if contours:
            for c in contours:
                area = int(cv2.contourArea(c))
                if area >> 5 :
                    (x,y),radius = cv2.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(result,center,radius,green,2)
                
                
                
    ## repeat the previous steps for red            
    r_components, output, stats, centroids = cv2.connectedComponentsWithStats(imgr)
    for i in range(1, r_components):
        x = int(centroids[i,0]) 
        y = int(centroids[i,1]) + 20
        cv2.putText(result,'Red Buoy', (x,y), font, fontScale, red, thickness, cv2.LINE_AA )
        mask = cv2.dilate(imgr,kernel,iterations=1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        imgray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours, hier = cv2.findContours(thresh, 1, 2)
        radius_r = []
        if contours:
            for c in contours:
                area = int(cv2.contourArea(c))
                if area >> 5 :
                    (x,y),radius = cv2.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(result,center,radius,red,2)
                
                
    # Write the Video            
    out.write(result)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
       
        

cap.release()
out.release()

