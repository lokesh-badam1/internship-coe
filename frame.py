import cv2
import os

cam = cv2.VideoCapture("illegal.mp4")

frameno = 0
while(True):
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = str(frameno) + 'illegal.jpg'
      print ('new frame captured...' + name)

      cv2.imwrite(name, frame)
      frameno += 1
      if frameno == 2:
         break
   else:
      break

cam.release()
cv2.destroyAllWindows()
