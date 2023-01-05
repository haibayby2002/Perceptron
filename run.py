from Perceptron.LaneDetection import LaneDetection

# x = LaneDetection(img='example.png')

# x.follow_lane()

import cv2
import numpy as np
image_url = "video/video1.mp4"       #320x240

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    # triangle = np.array([[
    # (200, height),
    # (800, 350),
    # (1200, height),]], np.int32)
    triangle = np.array([[
    (1, 200),
    (80, 1),
    (240, 1),
    (320, 200),
    ]], np.int32)
    # triangle = np.array([[(160, 130), (350, 130), (250, 300)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    cv2.imshow("", mask)
    # cv2.waitKey(0)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
 
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
                # print("left: " + str((slope, intercept)))
            else:
                right_fit.append((slope, intercept))
                # print("right: " + str((slope, intercept)))
            
    # print(len(left_fit), len(right_fit))
    if len(left_fit) == 0:
        left_fit_average = np.array([right_fit[0][0], 30])
    else:
        left_fit_average  = np.average(left_fit, axis=0)

    if len(right_fit) == 0:
        right_fit_average = np.array([left_fit[0][0], -90])
    else:
        right_fit_average  = np.average(right_fit, axis=0)
    print("Left: ", left_fit_average, "Right: " ,right_fit_average)
    # right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

cap = cv2.VideoCapture(image_url)


# Normal
# while(cap.isOpened()):
#     _, frame = cap.read()

#     canny_image = canny(frame)
#     cropped_canny = region_of_interest(canny_image)

#     lines = houghLines(cropped_canny)
#     averaged_lines = average_slope_intercept(frame, lines)
#     line_image = display_lines(frame, averaged_lines)
#     combo_image = addWeighted(frame, line_image)
#     cv2.imshow("result", combo_image)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# OOP
while(cap.isOpened()):
    _, frame = cap.read()
    lane = LaneDetection(frame)
    canny_image = lane.canny()
    cropped_canny = lane.region_of_interest(canny_image)

    lines = lane.houghLines(cropped_canny)
    averaged_lines = lane.average_slope_intercept(lines)
    line_image = lane.display_lines(averaged_lines)
    combo_image = lane.addWeighted(line_image)
    cv2.imshow("result", combo_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()