from pylab import *
from time import sleep
import serial
import matplotlib.pyplot as plt
ser = serial.Serial('COM4', 9600) # Establish the connection on a specific port
counter = 0
plt.ion()
time = 0.05
tempx=0
while True:
     str=int(ser.readline()) # Read the newest output from the Arduino
     if(counter==0):
         print "x=%d" %(str)
         """pltz.plot([time-1,time],[tempz,str])
         tempz=str
         pltz.ylabel('x acceleration')
         pltz.xlabel('Time')
         draw()"""
     elif(counter==1):
         print "y=%d" %(str)
         """plty.plot([time-1,time],[tempy,str])
         tempy=str
         plty.ylabel('y acceleration')
         plty.xlabel('Time')
         draw()"""
     else:
         print "z=%d" %(str)
         plt.plot([time-0.05,time],[tempx,str])
         tempx=str
         plt.ylabel('z acceleration')
         plt.xlabel('Time')
         draw()
     counter+=1
     if(counter>=3):
         counter=0
         time+=0.05
         sleep(.05)
     if (time-int(time))==0:
          plt.close()
          plt.ion()
     
      # Delay for one tenth of a second
     """if counter == 255:
     counter = 32"""
