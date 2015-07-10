import cv2
import cv2.cv
import numpy as np
import sys
import matplotlib
##matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from recog import *

cap = cv2.VideoCapture(0)
t,cx1,cy1=0,0,0
plt.ylabel('Y')
plt.xlabel('X')
##fig = plt.Figure()
##ax2 = fig.add_subplot(1,1,1)
plt.ylim((-480,0))
plt.xlim((0,640))
plt.ion()

while(1):

    # Take each frame
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.medianBlur(frame,5)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_orange = np.array([2, 150, 100])
    upper_orange = np.array([27,255,255])

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    blur = cv2.GaussianBlur(res,(5,5),0)
    blur2 = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(blur,100,200)
    ret3,th3 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        print "No object found to trace! Exiting..."
        plt.axis('off')
        plt.savefig('fig.png',bbox_inches='tight')
        plt.close()
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
    max_index = np.argmax(areas)
    cnt=contours[max_index]

##    x,y,w,h = cv2.boundingRect(cnt)
##    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(frame,ellipse,(0,255,0),2)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.rectangle(frame, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 0, 255), -1)
##    cv2.drawContours(frame,[cnt],0,(255,0,0),-1)

    k = cv2.waitKey(1) 
    if k == -1:
##        k = cv2.waitKey(1)
##        if(k!=32):
        plt.plot([cx1,cx1],[-cy1,-cy1],'b-')
        line = plt.plot([cx-.1,cx],[-cy+.1,-cy],'ro')
        plt.draw()
##        plt.cla()
##        line[0].set_visible(False)
        line[0].remove()
        del line
        plt.ylim((-480,0))
        plt.xlim((0,640))
        cv2.imshow('frame',frame)
        t=0
    if k == 27:
        plt.axis('off')
        plt.savefig('fig.png',bbox_inches='tight')
        plt.close()
        break
    if k==32 and t>0:
        cv2.imshow('frame',frame)
        plt.plot([cx1,cx],[-cy1,-cy],'b-')
        line = plt.plot([cx-.1,cx],[-cy+.1,-cy],'ro')
        plt.draw()
        line[0].remove()
        del line
##        l = ax2.plot([cx-.1,cx],[-cy+.1,-cy],'ro')
##        l.pop(0).remove()
##        plt.ylim((-480,0))
##        plt.xlim((0,640))
        
    t=1
    cx1,cy1=cx,cy
##    cv2.imshow('mask',mask)
##    cv2.imshow('res',res)
##    cv2.imshow('blur',blur)
##    cv2.imshow('otsu',th3)
##    cv2.imshow('a',img)
##    cv2.imshow('edges',edges)
    
cap.release()
cv2.destroyAllWindows()

recognize('fig.png')

