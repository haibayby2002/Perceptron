import cv2
import numpy as np
# from SceneUnderstanding import SceneUnderstanding

# image_url = "video/test1.mp4"    #1280x720
image_url = "video/video1.mp4"       #320x240
# image_url = "video/video2.mp4"         #320x240


class LaneDetection:
    def __init__(self, img, roi = np.array([[
    (1, 200),
    (80, 1),
    (240, 1),
    (320, 200),
    ]], np.int32)):

        # super().__init__(img)
        self.img = img
        self.roi = roi
        self.left_lane = 0
        self.right_lane = 0



    def follow_lane(self):
        pass

    # Return slope and intercept 
    def get_left_lane(self):
        return self.left_lane

    # Return slope and intercept
    def get_right_lane(self):
        return self.right_lane

    def addWeighted(self, line_image):
        return cv2.addWeighted(self.img, 0.8, line_image, 1, 1)

    def canny(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        kernel = 5
        blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
        canny = cv2.Canny(gray, 50, 150)
        return canny
 
    def region_of_interest(self, canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)
        triangle = self.roi
        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        cv2.imshow("", mask)
        return masked_image

    def houghLines(self, cropped_canny):
        return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
            np.array([]), minLineLength=40, maxLineGap=5)

    def display_lines(self,lines):
        line_image = np.zeros_like(self.img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
                    pass
        return line_image
    
    def make_points(self, line):
        slope, intercept = line
        y1 = int(self.img.shape[0])
        y2 = int(y1*3.0/5)      
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return [[x1, y1, x2, y2]]
    
    def average_slope_intercept(self, lines):
        left_fit    = []
        right_fit   = []
        if lines is None:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if abs(slope) < 1e-3:
                    continue
                if slope < 0: 
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

        print("Left: ", left_fit, "Right: ", right_fit)
                
        if len(left_fit) == 0:
            left_fit_average = np.array([right_fit[0][0], 30])
        else:
            left_fit_average  = np.average(left_fit, axis=0)

        if len(right_fit) == 0:
            right_fit_average = np.array([left_fit[0][0], -90])
        else:
            right_fit_average  = np.average(right_fit, axis=0)

        # print("Left: ", left_fit_average, "Right: " ,right_fit_average)

        left_line  = self.make_points(left_fit_average)
        right_line = self.make_points(right_fit_average)
        averaged_lines = [left_line, right_line]

        # print("Avg: ", averaged_lines)
        return averaged_lines