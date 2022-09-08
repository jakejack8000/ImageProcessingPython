import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep
## Define Fonts and colors for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
orange = (0,69,255)
thickness = 1
#read video
cap = cv2.VideoCapture(r'Gate_1r.mp4')
#cap = cv2.VideoCapture(r'Gate_2.mp4')

#Get video width and height and define video writer codec
frame_height = int(cap.get(4))
frame_width = int(cap.get(3))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outgate2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# define kernels for morphological operations, first one is vertical
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
while True:
    #Read frame
    ret, img = cap.read()
    #Create a copy for output
    edges = img.copy()
    #Convert to Grayscale
    imgo = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
    #Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Increase Contrast
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    imgo = clahe.apply(imgo)
    
    #Morphological erosion with vertical kernel to keep only the gate
    imgo = cv2.erode(imgo,kernel,iterations = 1)
    imgo = cv2.dilate(imgo,kernel,iterations = 1)
    imgo = cv2.erode(imgo,kernel,iterations = 1)
    imgo = cv2.dilate(imgo,kernel2,iterations = 1)

    #Keep only the highest pixels
    ret, imgo = cv2.threshold(imgo, 235, 255, 0)
  
    #create mask
    mask = cv2.dilate(imgo,kernel,iterations=1)
    #pass the mask on original frame

    masked = cv2.bitwise_and(edges, edges, mask=mask)
    # Convert to grayscale and threshold
    imgray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    #find minimum enclosing rectangle of contours, draw it only if higher than specific area
    contours, hier = cv2.findContours(thresh, 1, 2)
    if contours:
            for c in contours:
                area = int(cv2.contourArea(c))
                if area >> 10 :
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(edges,[box],0,orange,2)
                    x = int(rect[0][0])
                    y = int(rect[0][1])
                    c_number= cv2.putText(edges,'Orange Gate', (x,y), font, fontScale, orange, thickness, cv2.LINE_AA )                    

    out.write(masked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()