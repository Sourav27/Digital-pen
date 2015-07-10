from serial import *
from time import sleep
import matplotlib,time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

usb = Serial('COM4', 9600)
#usb.timeout = 1
counter=0
x,y,z=0,0,0
tstart=time.time()
t=tstart
plt.ylabel('X acc')
plt.xlabel('time')
plt.ion()
##plt.show()
from time import sleep
##fig = plt.figure()
##ax = fig.gca(projection='3d')
x1,y1,z1=0,0,0
##ax.set_xlabel('X axis')
##ax.set_ylabel('Y axis')
##ax.set_zlabel('Z axis')
##plt.ion()
##plt.show()
while True:
    str1=float(usb.readline())
    if counter==0:
        x=str1
##        x+=str1/2*(time.time()-t)*(time.time()-t)
        print "x=%f"%(x)
        plt.plot([t,time.time()],[x1,x])
        plt.draw()
        sleep(0.02)
        x1=x
    elif counter==1:
        y=str1
##        y+=str1/2*(time.time()-t)*(time.time()-t)
##        print "y=%f"%(y)
    else:
        z=str1
##        z+=str1/2*(time.time()-t)*(time.time()-t)
##        print "z=%f"%(z)
##        ax.plot([x1,x], [y1,y], [z1,z],'b-')
##        plt.draw()
##        sleep(0.02)
##        x1=x
##        y1=y
##        z1=z
    t=time.time()
    counter+=1
    if(counter>2):
        counter=0
#print 'FPS:' , 1000/(time.time()-tstart)
        
