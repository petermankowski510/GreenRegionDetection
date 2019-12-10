# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:43:34 2019

@author: Peter Mankowski
"""
import cv2
import numpy as np
import itertools
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy
from PIL import Image

# Variable definitions
obj_volume = 1000

cap = cv2.VideoCapture(0)

"""RGB limits to sense GREEN"""
#lw,lw1,lw2=60,90,150
#hw,hw1,hw2=80,150,255
lw,lw1,lw2=49,90,150
hw,hw1,hw2=83,150,255

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_green = np.array([lw,lw1,lw2])
    upper_green = np.array([hw,hw1,hw2])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    """kernel to watch through the image, or the frame, 
        and dilated to smooth the image"""
    kernel = np.ones((5,5),'int')
    
    """The first parameter is the original image, kernel is the matrix with which image is  
        convolved and third parameter is the number of iterations, which will determine how much  
        you want to erode/dilate a given image"""
    dilated = cv2.dilate(mask,kernel, iterations=1)    
    erosion = cv2.erode(mask, kernel, iterations=1)    
    
    #cv2.imshow('dilated',dilated)
    #cv2.imshow('erosion',erosion)
    
    edges = cv2.Canny(frame,100,200)    
    #cv2.imshow('Edges',edges)
    
    numpy_horizontal = np.hstack((dilated, erosion, edges))
    cv2.imshow('Numpy_horizontal', numpy_horizontal)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)
    
    """The first argument is the source image, which should be a grayscale image. 
        The second argument is the threshold value which is used to classify the pixel values. 
        The third argument is the maximum value which is assigned to pixel values exceeding the threshold
        Threshold value which is used to classify the pixel values. 
        Maximum value which is assigned to pixel values exceeding the threshold"""
    thrs_toggle = 127 # We had 3 originally
    thrs_max = 255
    
    ret,thrshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),thrs_toggle,thrs_max,cv2.THRESH_BINARY)
    
    """first one is source image, second is contour retrieval mode, third is contour approximation method 
        Outputs the contours and hierarchy values"""
    contours,hier = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    """To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        Untested block"""    
    contours_frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('contours_frame',contours_frame)
    #print(contours)
       
    for cnt in contours:
        #Contour area is taken
        area = cv2.contourArea(cnt)
        """It is not necessary to watch for all the contour areas because much smaller ones are annoying.
         Hence,  we define a minimum limit for the area. Here I will take any area over 1000. 
         If an area over obj_volume is detected then we put a text on the frame and  draw a rectangle around it"""
        if area >obj_volume:
            print(len(contours))
            #x2=min(min([[j[0][0] for j in i] for i in contours]))
            #x1=min(min([[j[0][0] for j in i] for i in contours]))
            #y2=max(max([[j[0][1] for j in i] for i in contours]))
            #y1=min(min([[j[0][1] for j in i] for i in contours]))
            temp=[[j[0][0] for j in i] for i in contours]
            x=np.mean(list(itertools.chain.from_iterable(temp)))
            dX=np.std(list(itertools.chain.from_iterable(temp)))
            temp=[[j[0][1] for j in i] for i in contours]
            # best fit of data
             
            temp11=[[j[0][0] for j in i] for i in contours]
            #xFar=np.mean(list(itertools.chain.from_iterable(temp11)))
            xF=list(itertools.chain.from_iterable(temp11))
   
            yF=list(itertools.chain.from_iterable(temp))

            (mu, sigma) = norm.fit(yF)
            (mu1, sigma1) = norm.fit(xF)
            # the histogram of the data
            #n, bins, patches = plt.hist(xF, 60, normed=1, facecolor='green', alpha=0.75)
            #dY=1.*sigma
            yFar=scipy.stats.norm.ppf(0.05,mu,sigma)
            xFar=scipy.stats.norm.ppf(0.05,mu1,sigma1)
            dX=abs(x-xFar)
            
            y=np.mean(list(itertools.chain.from_iterable(temp)))
            dY=abs(y-yFar)
            #dY=np.std(list(itertools.chain.from_iterable(temp)))
            
            img=Image.fromarray
            print(x,y,dX,dY)
            
            
            cv2.rectangle(frame,(int(x-dX),int(y-dY)),(int(x+dX),int(y+dY)),(255,0,0),3)
            
            # Conversion from OpenCV to Image PIL            
            rect = frame[int(y-dY):int(y+dY),int(x-dX):int(x+dX)]
            
            avG=np.mean(rect[:,:,0])
            avR=np.mean(rect[:,:,1])
            avB=np.mean(rect[:,:,2])
            
            img1=Image.fromarray(rect)
            
            """RGB Values dump"""            
            print("Average Blue", avB)
            print("Average Red", avR)
            print("Average Green", avG)
            
            """Pick one pixel at the center and display its RGB-(debugging tool)"""
            print("One pixel Blue :",rect[int(dY),int(dX),0])
            print("One pixel Green:",rect[int(dY),int(dX),1])
            print("One pixel Red  :",rect[int(dY),int(dX),2])
            
            #plt.show()
            cv2.imshow("SEEEEE",rect)
            print("#####################")
            
            cv2.polylines(frame,[cnt],True,(0,0,255))
            cv2.putText(frame, "Green Object Detected", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.rectangle(frame,(5,40),(400,100),(0,255,255),2)
    
    cv2.imshow('frame',frame)
    # Break the loop as before using ESC key
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
