#!/usr/bin/python

# troll.py

# Face Detection using OpenCV. Replaces faces with troll faces.
# Based on sample code from: http://python.pastebin.com/m76db1d6b

# Usage: python trolls.py <image_file> <output_file>

import sys, os, random
from opencv.cv import *
from opencv.highgui import *

def detectObjects(image, path):
  """Converts an image to grayscale and prints the locations of any 
     faces found"""
  grayscale = cvCreateImage(cvSize(image.width, image.height), 8, 1)
  cvCvtColor(image, grayscale, CV_BGR2GRAY)

  storage = cvCreateMemStorage(0)
  cvClearMemStorage(storage)
  cvEqualizeHist(grayscale, grayscale)
  cascade = cvLoadHaarClassifierCascade(
    '/home/videator/Documents/haarcascade_frontalface_alt.xml',
    cvSize(1,1))
  faces = cvHaarDetectObjects(grayscale, cascade, storage, 1.2, 2,
                             CV_HAAR_DO_CANNY_PRUNING, cvSize(50,50))

  if faces:
    counter = 0
    for f in faces:
      counter += 1
      #cvRectangle(image, (f.x, f.y), (f.x+f.width, f.y+f.height), CV_RGB(255, 0, 0), 3, 8, 0)
      current_troll = trolls[counter % len(trolls)]
      scaling_factor = 1.5
      scaled_height, scaled_width = int(current_troll.rows/float(current_troll.cols)*f.width*scaling_factor), int(f.width*scaling_factor)
      small_current_troll = cvCreateMat(scaled_height, scaled_width, CV_8UC3)
      cvResize(current_troll, small_current_troll)
      roi = image[f.y-f.height/4:f.y+small_current_troll.rows-f.height/4, f.x-f.width/4:f.x+small_current_troll.cols-f.width/4]
      cvCopy(small_current_troll, roi)
      cvSaveImage(path, image)
      print("[(%d,%d) -> (%d,%d)]" % (f.x, f.y, f.x+f.width, f.y+f.height))

def main():
  image = cvLoadImage(sys.argv[1]);
  troll = cvLoadImage('trolls/troll.jpg')
  megusta = cvLoadImage('trolls/megusta.jpg')
  youdontsay = cvLoadImage('trolls/youdontsay.jpg')
  genius = cvLoadImage('trolls/genius.jpg')
  global trolls
  trolls = [genius, youdontsay, troll, megusta]
  random.shuffle(trolls)
  detectObjects(image, sys.argv[2])

if __name__ == "__main__":
  main()
