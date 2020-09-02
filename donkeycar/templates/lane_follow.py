# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:05:25 2020

@author: Isbla
"""

import cv2
import numpy as np

def canny(image):
   gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blur, 50, 100)
   return edges

def region(image):
   polygons = np.array([[(0, 250), (0, 200), (150, 100), (500, 100), (650, 200), (650, 250)]])
   mask = np.zeros_like(image)
   cv2.fillPoly(mask, polygons, 255)
   masked_image = cv2.bitwise_and(image, mask)
   return masked_image

def disp_lines(image, lines):
   line_img = np.zeros_like(image)
   if lines is not None:
       for line in lines:
           x1, y1, x2, y2 = line.reshape(4)
           cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 5)
   return line_img
def avg_lines(image, lines):
   left = []
   right = []
   for line in lines:
       x1,y1,x2,y2 = line.reshape(4)
       params = np.polyfit((x1,x2),(y1,y2),1)
       slope = params[0]
       y_int = params[1]
       if slope < 0:
           left.append([slope,y_int])
       if slope > 0:
           right.append([slope,y_int])
   left_avg = np.average(left, axis = 0)
   right_avg = np.average(right, axis = 0)
   left_line = make_coords(image, left_avg)
   right_line = make_coords(image, right_avg)
   return np.array([left_line,right_line])

def make_coords(image, line_params):
   slope, intercept = line_params
   lower_y = 300
   upper_y = 120
   lower_x = int(lower_y - intercept/slope)
   upper_x = int(upper_y - intercept/slope)
   return np.array([lower_x,upper_x,lower_y,upper_y])

img = cv2.imread("img3.jpg")
lane_img = np.copy(img)
lane_img = (cv2.resize(lane_img, (650, 500)))
canny_img = canny(lane_img)
cropped = region(canny_img)
lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 50, np.array([]), minLineLength = 40, maxLineGap = 5)
display = disp_lines(lane_img, lines)

cv2.imshow("edges", display)
cv2.waitKey(0)
