# ------------
# @author   Jordan Reed
# @date     6/21/23
# @brief    created a class to get a roi on an image.
#
# ------------

import cv2
from copy import copy
import numpy as np

class Shape:
    def __init__(self, image, points = None):
        if points is None:
            self.points = []
        else:
            self.points = points
        print(points)
        self.image = copy(image)
        self.win_name = "ROI Selection"
        self.shapeDone = False
        self.color = (0,0,0)

    def draw_line(self, image, p1, p2):
        cv2.line(image, p1, p2, self.color, 3)
    
    def get_point(self, event, x, y, flags, parameters):
        # print('mouse detected')
        if event == cv2.EVENT_LBUTTONUP and not self.shapeDone:
            print(f'button clicked at {x},{y}')
            p = [x,y]
            if len(self.points) != 0:
                self.draw_line(self.image, self.points[-1], p)
                print('draw line called')
            self.points.append(p)
            print('point appended to list')
        elif event == cv2.EVENT_RBUTTONDOWN and not self.shapeDone:
        # elif event == cv2.EVENT_MOUSEHWHEEL and not self.shapeDone:
            print('Shape done right mouse click')
            self.draw_line(self.image, self.points[0], self.points[-1])
            self.shapeDone = True
        
    def get_roi(self):
        # temp = sf.copy(image)
        # print('created copy of image')

        if len(self.points) != 0:
            for i in range(len(self.points)-1):
                print('has points already')
                self.draw_line(self.image, self.points[i], self.points[i+1])

        # create window and mouse callback
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.get_point)

        while True:
            cv2.imshow(self.win_name, self.image)
            key = cv2.waitKey(1)

            if key == ord('c'):
                cv2.destroyAllWindows()
                break
            elif key == ord('d'):
                print('d pressed')
                self.draw_line(self.image, self.points[0], self.points[-1])
                self.shapeDone = True
    
    def apply_roi_mask(self, image):
        temp = copy(image)

        pts = sf.np.array(self.points)
        cv2.fillConvexPoly(temp, pts, self.color)
        mask = temp != 0
        img[mask] = 0

    def __str__(self):
        output = ''
        for p in self.points:
            output += f'({p[0]},{p[1]}) '
        return output
    
# testing code

import seg_functions as sf

pcna_stack, dapi_stack = sf.open_nd2file('nd2_files/6dpi-uoi2500Tg-2R-#17-sxn4003.nd2')

img = sf.compress_stack(pcna_stack)
img /= np.max(img)

# roi = Shape(img, [[0,0],[20,200],[850,800]])
roi = Shape(img)
# get roi on img
roi.get_roi()

roi.apply_roi_mask(img)

cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
cv2.imshow('Final', img)
cv2.waitKey(0)

print(roi)
