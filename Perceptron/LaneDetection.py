import cv2
import numpy as np
import os
import math

# from SceneUnderstanding import SceneUnderstanding

# image_url = "video/test1.mp4"    #1280x720
# image_url = "video/video1.mp4"       #320x240
# image_url = "video/video2.mp4"         #320x240



class LaneDetection:
    index = 0
    def __init__(self, img, roi = np.array([[
    (1, 200),
    (80, 1),
    (240, 1),
    (320, 200),
    ]], np.int32), width = 320, height = 240):

        # super().__init__(img)
        self.img = img
        self.roi = roi
        self.left_lane = 0
        self.right_lane = 0
        self.recommended_angel = 0  #Go ahead
        self.slope = 0
        self.width = width
        self.height = height
        self.step = 50      #for debugging



    def follow_lane(self):
        canny_image = self.canny()
        cropped_canny = self.region_of_interest(canny_image)

        lines = self.houghLines(cropped_canny)
        averaged_lines = self.average_slope_intercept(lines)
        line_image = self.display_lines(averaged_lines)
        combo_image = self.addWeighted(line_image)
        cv2.putText(combo_image, 
                "{:.2f}".format(self.slope) + ' ' + str(self.recommended_angel / np.pi * 180), 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
        cv2.imshow("result", combo_image)
        cv2.imshow("line", line_image)
        # Todo: based on the necessary function: call it here
        # decision_making = DecisionMaking(Perceptron (self))
        # decision_making.decide('lane_following', left, righ)
        # return self.addWeighted(line_image)
        angle = self.recommended_angel / math.pi * 180
        if abs(angle) > 20:
            return 'p.a' if angle < 0 else 'p.d'
        return 'p.r'

    # This should belongs to Support Utils
    # The input slope MUST FOLLOW THE STANDARD AXIS (multiply with -1 if it is image cooridnates)
    # The return value is radian
    @staticmethod
    def calculate_steering_angle(slope):
        # x = math.atan2(slope, 1)
        # if slope < 0:
        #     return -(math.pi / 2 + x)
        # return math.pi/2-x
        return math.atan2(slope, 1)
        

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
        kernel = 3  #change 5->3
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
        cv2.imshow("Mask", mask)
        cv2.imshow("Masked image", masked_image)
        return masked_image

    def houghLines(self, cropped_canny):
        return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
            np.array([]), minLineLength=20, maxLineGap=5)

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
                    slope =1e-3
                if slope < 0: 
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

        # print("Left: ", left_fit, "Right: ", right_fit)
                
        if len(left_fit) == 0:
            # path = r"log_image\lack_left" + str(LaneDetection.index) + ".jpg"
            # path = os.path.join(os.path.expanduser('~'),'log_image','lack_left'+ str(LaneDetection.index) + ".jpg")
            # cv2.imwrite(path, self.img)
            # LaneDetection.index+=1
            left_fit_average = np.array([right_fit[0][0], 30])
        else:
            left_fit_average  = np.average(left_fit, axis=0)

        if len(right_fit) == 0:
            # path = r"log_image\lack_right" + str(LaneDetection.index) + ".jpg"         
            # path = os.path.join(os.path.expanduser('~'),'log_image','lack_right'+ str(LaneDetection.index) + ".jpg")   
            # cv2.imwrite(path, self.img)
            # LaneDetection.index+=1
            right_fit_average = np.array([left_fit[0][0], -90])
        else:
            right_fit_average  = np.average(right_fit, axis=0)

        # print("Left: ", left_fit_average, "Right: " ,right_fit_average)

        # left_slope = left_fit_average[0]

        # cv2.imshow("Line")

        left_line  = self.make_points(left_fit_average)
        right_line = self.make_points(right_fit_average)
        averaged_lines = [left_line, right_line]

        
        # Solve linear equation
        # if left_fit_average[0] == right_fit_average[0]:
        #     self.slope = left_fit_average[0]
        # else:
        #     _a = np.array([[-left_fit_average[0], 1], [-right_fit_average[0], 1]])
        #     _b = np.array([left_fit_average[1], right_fit_average[1]])
        #     _x = np.linalg.solve(_a,_b)
        #     fixed_point = [self.width / 2, self.height]
        #     # print(_x)
            

        #     _a = np.array([[_x[0], 1], [fixed_point[0], 1]])
        #     _b = np.array([_x[1], fixed_point[1]])
        #     line_sol = np.linalg.solve(_a, _b)
        #     self.slope = line_sol[0]
        #     print(line_sol)

        # print("Avg: ", averaged_lines)
        # avg = averaged_lines[0][0]
        # print(left_line)

        # print("left_fit_average: ", left_fit_average, "right_fit_average: ", right_fit_average)
        # Calculate slope
        left_slope = left_fit_average[0]    
        right_slope = right_fit_average[0]
        
        if left_slope == 0 or right_slope == 0: #vertical line
            left_slope = 1
            right_slope = -1
        average_slope = (1/left_slope + 1/right_slope) / 2
        
        
        self.slope = -average_slope
        self.recommended_angel = LaneDetection.calculate_steering_angle(self.slope)
        # print("Slope: ", self.slope, "Angle: ",self.recommended_angel / math.pi * 180)
        print("Slope: ", self.slope, "Angle: ",math.degrees(self.recommended_angel))
        # print(left_slope, right_slope,self.recommended_angel / np.pi * 180)

        LaneDetection.index+=1
        if LaneDetection.index == self.step:
            LaneDetection.index = 0

        print("left: ", left_slope, "right: ", right_slope)

        return averaged_lines