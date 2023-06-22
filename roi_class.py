# ------------
# @author   Jordan Reed
# @date     6/21/23
# @brief    created a class to get a roi on an image.
#
# ------------

import cv2
from copy import copy
import numpy as np

class ROI:
    """
    Generates a list of points to create a region of interest in an image. Can also apply the mask to an image.
    """
    def __init__(self, image, points = None):
        if points is None:
            self.points = []
        else:
            self.points = points

        self.image = np.array(image)        # copy of image to create an roi on
        self.win_name = "ROI Selection"     # name of window when capturing coordinates
        self.shapeDone = False              # is the shape closed
        self.color = (0,0,0)                # color to draw the lines and fill the polygon with
        self.mask = None                    # roi binary mask

    def draw_line(self, image, p1, p2):
        """
        Draws a line on an image from point 1 to point 2.

        :param image: image to draw line on, numpy array preferred
        :param p1: point in coordinate form, either tuple or list
        :param p2: point in coordinate form, either tuple or list
        """
        cv2.line(image, p1, p2, self.color, 3)
    
    def get_point(self, event, x, y, flags, parameters):
        """
        Callback for a mouse event using OpenCV. On a left click, create a point and draw a line. On a right click, connect the shape.

        :param event: mouse event, opencv mouse event
        :param x: x coordinate
        :param y: y coordinate
        :param flags: opencv var
        :param parameters: opencv var
        """
        if event == cv2.EVENT_LBUTTONUP and not self.shapeDone:
            # create point
            p = [x,y]

            # if not first point, draw a line between current point and last point
            if len(self.points) != 0:
                self.draw_line(self.image, self.points[-1], p)

            # add point to end of list
            self.points.append(p)
        elif event == cv2.EVENT_RBUTTONDOWN and not self.shapeDone:
            # close the shape by connecting first and last points
            self.draw_line(self.image, self.points[0], self.points[-1])
            self.shapeDone = True
        
    def get_roi(self):
        """
        Uses OpenCV's GUI to capture coordinates from user to create an ROI.
        """
        # draw starting points (if any) on image
        if len(self.points) != 0:
            for i in range(len(self.points)-1):
                self.draw_line(self.image, self.points[i], self.points[i+1])

        # create window and mouse callback
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.get_point)

        # print instructions
        print("\nLeft click to select a point. Right click or type 'd' to connect your last point to the first point.")
        print("Type 'c' to exit and move to the next step.")

        # update window
        while True:
            cv2.imshow(self.win_name, self.image)
            key = cv2.waitKey(1)

            if key == ord('c'):
                cv2.destroyAllWindows()
                break
            elif key == ord('d'):
                # close shape
                self.draw_line(self.image, self.points[0], self.points[-1])
                self.shapeDone = True
    
    def apply_roi_mask(self, image):
        """
        Create a polygon shape from the coordinate points and use that shape to apply a binary mask to the image provided

        :param image: image to apply mask to, numpy preferred, MUST be same shape as mask/original image
        """
        # test = np.array(image)
        temp = np.array(image)
        if temp.shape != self.image.shape:
            print("Image must be the same size as the ROI mask and the original image.")
            return
        
        # if no mask created yet, create one from points
        if self.mask is None:
            pts = np.array(self.points)
            cv2.fillConvexPoly(temp, pts, self.color)
            self.mask = temp != 0

        # apply mask to original image
        image[self.mask] = 0

    def __str__(self):
        output = ''
        for p in self.points:
            output += f'({p[0]},{p[1]}) '
        return output

# -------------------
# testing code

# import seg_functions as sf

# pcna_stack, dapi_stack = sf.open_nd2file('nd2_files/6dpi-uoi2500Tg-2R-#17-sxn4003.nd2')

# img = sf.compress_stack(pcna_stack)
# img /= np.max(img)

# # roi = Shape(img, [[0,0],[20,200],[850,800]])
# roi = ROI(img)
# # get roi on img
# roi.get_roi()

# roi.apply_roi_mask(img)

# cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
# cv2.imshow('Final', img)
# cv2.waitKey(0)

# print(roi)
