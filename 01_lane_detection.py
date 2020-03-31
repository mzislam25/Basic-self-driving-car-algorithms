import cv2
import numpy as np
import math
import sys

global first_frame

first_frame = 1
cap = cv2.VideoCapture('vid/videoplayback.mp4')

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def make_coordinate(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None: 
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            if all(x < 5000 for x in (x1,y1,x2,y2)):
	            cv2.line(line_image, (x1,y1), (x2,y2),(0,255,0), 10)
    return line_image

while True:
    _, frame75 = cap.read()
    #frame75 = rescale_frame(frame, percent=50)
    image = np.copy(frame75)
    gray_image = cv2.cvtColor(frame75, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(frame75, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(gray_image, 200, 255)
    com_mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask_image = cv2.bitwise_and(gray_image, com_mask)

    blur_image = cv2.GaussianBlur(mask_image, (5, 5), 0)

    thres_low = 50
    thres_high = 150
    canny_image = cv2.Canny(blur_image, thres_low, thres_high)

    imshape = frame75.shape
    lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
    top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
    top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

    mask = np.zeros_like(canny_image)   
    if len(canny_image.shape) > 2:
        channel_count = canny_image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    roi_image = cv2.bitwise_and(canny_image, mask)

    lines = cv2.HoughLinesP(roi_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    #line_image = display_lines(image, lines)

    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2), deg=1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_line = [214,720,636,432]
    right_line = [1160,720,672,432]
    
    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        left_line = make_coordinate(image, left_fit_avg)
    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)  
        right_line = make_coordinate(image, right_fit_avg)

    avg_line = np.array([left_line, right_line])

    line_image = display_lines(image, avg_line)

    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
##    cv2.imshow('frame',frame75)
##    cv2.imshow('gray',gray_image)
##    cv2.imshow('mask',mask_image)
##    cv2.imshow('blur',blur_image)
##    cv2.imshow('canny',canny_image)
##    cv2.imshow('line',line_image)
    cv2.imshow('combo',combo_image)
##    cv2.imshow('res',res)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
