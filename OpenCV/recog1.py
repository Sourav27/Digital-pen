import numpy as np
import cv2
import os
import sys
from common import mosaic
from digits import *
from collections import namedtuple
import operator
from matplotlib import pyplot as plt

classifier_fn = 'digits_svm.dat'
if not os.path.exists(classifier_fn):
    print '"%s" not found, run digits.py first' % (classifier_fn)
    sys.exit()
model = SVM()
model.load(classifier_fn)

class Digit():
    def __init__(self, value, x, y):
        self.value = value
        self.x = x
        self.y = y

d = []
i,t=0,0
img = cv2.imread('fig.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
bin = cv2.medianBlur(bin, 3)

ctrs, hier = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    
cv2.drawContours(img,ctrs,0,(255,0,0),-1)
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect,ctr in zip(rects,ctrs):
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    lengy = int(rect[3] * 1.1)
    lengx = int(rect[2] * 1.1)
    pt1 = int(rect[1] + rect[3] // 2 - lengy // 2)
    pt2 = int(rect[0] + rect[2] // 2 - lengx // 2)
    roi = bin[pt1:pt1+lengy, pt2:pt2+lengx]
    cv2.rectangle(img, (pt2,pt1),(pt2+lengx,pt1+lengy), (0, 0, 255), 3)
    if cv2.contourArea(ctr)<50:
        continue
    # Resize the image
    roi = cv2.resize(roi, None,fx=.2,fy=.2, interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
##    s = 1.5*float(pt2+leng)/SZ
##    m = cv2.moments(roi)
##    c1 = np.float32([m['m10'], m['m01']]) / m['m00']
##    c0 = np.float32([SZ/2, SZ/2])
##    t = c1 - s*c0
##    A = np.zeros((2, 3), np.float32)
##    A[:,:2] = np.eye(2)*s
##    A[:,2] = t
##    bin_norm = cv2.warpAffine(roi, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
##    cv2.imshow('bin_norm',roi)
##    bin_norm = deskew(bin_norm)

    sample = preprocess_hog([roi])
    digit = model.predict(sample)[0]
##    print digit
    cv2.imshow('roi%d'%t,roi)
    t+=1
    m = cv2.moments(ctr) 
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    print "x = %f , y = %f \n" % (cx,cy)
    cv2.rectangle(img, (cx-2,cy-2),(cx+2,cy+2), (0, 0, 255), -1)
    d.append(Digit(digit,cx,cy))
    cv2.putText(img, '%d'%digit, (cx, pt1), cv2.FONT_HERSHEY_PLAIN, 3.0, (200, 0, 0), thickness = 1)

cv2.imshow('frame',bin)
cv2.imshow('result',img)

d.sort(key=operator.attrgetter('x'))
for a in d:
    print a.value
result=0
while (i < len(d)):
    result = result*10+d[i].value
    i+=1
print result
cv2.waitKey(0)
cv2.destroyAllWindows()
