#!/usr/bin/python

# troll.py

# Face Detection using OpenCV. Replaces faces with troll faces.
# Usage: python trolls.py <image_file> <output_file>

import sys, os, random, Image
from opencv.cv import *
from opencv.highgui import *

def detectObjects(image_path, path):
  """Converts an image to grayscale and prints the locations of any 
     faces found"""
  image = cvLoadImage(image_path);
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
    background = Image.open(image_path)
    counter = 0
    for f in faces:
      counter += 1
      current_troll = trolls[counter % len(trolls)]
      paste_troll_over_background(background, current_troll, (f.x, f.y), f.width)
      print("[(%d,%d) -> (%d,%d)]" % (f.x, f.y, f.x+f.width, f.y+f.height))
    background.save(path)

def paste_troll_over_background(background, troll, pos, face_width):
  troll_face, troll_mask, shift_factors = troll
  shift_x, shift_y = shift_factors
  troll_width, troll_height = troll_face.size
  scaling_factor = 1.5
  scaled_width = int(face_width*scaling_factor)
  scaled_height = int(troll_height/float(troll_width)*scaled_width)
  troll_face = troll_face.resize((scaled_width, scaled_height))
  troll_mask = troll_mask.resize((scaled_width, scaled_height))
  pos = (pos[0] - scaled_width/shift_x, pos[1] - scaled_height/shift_y)
  background.paste(troll_face, pos, troll_mask)

def main():
  global trolls
  trolls = []
  shift_factors = {'troll': (8, 8), 'megusta': (5, 5), 'yaoming': (20, 4)}
  for troll in shift_factors.keys():
    trolls.append((Image.open('trolls/'+troll+'.png'), Image.open('trolls/masks/'+troll+'.png'), shift_factors[troll]))
  random.shuffle(trolls)
  detectObjects(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
  main()
