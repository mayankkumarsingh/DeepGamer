import numpy as np
import math
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean
import numpy as np
import cv2
from image_processing_utils import *
from scipy import signal
from Line import Line, calc_curvature


class Lane_Detector:

    def __init__(self, n_frames=1, line_segments = 10, transform_offset=0):
        """
        Tracks lane lines on images or a video stream using techniques like Sobel operation, color thresholding and
        sliding histogram.
        :param perspective_src: Source coordinates for perspective transformation
        :param perspective_dst: Destination coordinates for perspective transformation
        :param n_frames: Number of frames which will be taken into account for smoothing
        :param cam_calibration: calibration object for distortion removal
        :param line_segments: Number of steps for sliding histogram and when drawing lines
        :param transform_offset: Pixel offset for perspective transformation
        """
        self.n_frames = n_frames
        self.line_segments = line_segments
        self.image_offset = transform_offset

        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.offset = 0.0

        self.dists = []


    def __line_plausible(self, left, right):
        """
        Determines if pixels describing two line are plausible lane lines based on curvature and distance.
        :param left: Tuple of arrays containing the coordinates of detected pixels
        :param right: Tuple of arrays containing the coordinates of detected pixels
        :return:
        """
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            return are_lanes_plausible(new_left, new_right)

    def __check_lines(self, left_x, left_y, right_x, right_y):
        """
        Compares two line to each other and to their last prediction.
        :param left_x:
        :param left_y:
        :param right_x:
        :param right_y:
        :return: boolean tuple (left_detected, right_detected)
        """
        left_detected = False
        right_detected = False

        if self.__line_plausible((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.__line_plausible((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.__line_plausible((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected

    def __draw_info_panel(self, img):
        """
        Draws information about the center offset and the current lane curvature onto the given image.
        :param img:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)

    def __draw_lane_overlay(self, img, capture_region):
        """
        Draws the predicted lane onto the image. Containing the lane area, center line and the lane lines.
        :param img:
        """
        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])

        # lane area
        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = BirdsEyePerspectiveInverse(mask, capture_region)

        overlay[mask == 1] = (255, 128, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        # center line
        mask[:] = 0
        mask = draw_poly_arr(mask, self.center_poly, 20, 255, 5, True, tip_length=0.5)
        mask = BirdsEyePerspectiveInverse(mask, capture_region)
        img[mask == 255] = (255, 75, 2)

        # lines best
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255)
        mask = BirdsEyePerspectiveInverse(mask, capture_region)
        img[mask == 255] = (255, 200, 2)

    def process_screengrab(self, orig_image, capture_region):

        # topleftx, toplefty, botrightx, botrighty = capture_region
        tlx, tly, brx, bry = capture_region
        w = brx - tlx
        h = bry - tly
        t = tly     # t = tly = 40
        b = bry     # b = bry = 520
        l = tlx     # t = tlx = 0
        r = brx     # r = brx = 640

        # 3rd person camera adjustments
        vertices=mask_vertices(l,t,b,r,w,h)
        roi_extracted_image = mask_image(orig_image,vertices)

        # Focussing on the road portion of the image
        # This depends on camera settings from game to game
        v_cutoff = np.int(0.35*h)
        mask_3ch = generate_lane_mask(roi_extracted_image, v_cutoff=v_cutoff)
        lane_masked_image = cv2.bitwise_and(roi_extracted_image, mask_3ch)
        # Convert to a binary mask
        mask = mask_3ch[:,:,0]

        mask_1ch =  mask * ((mask == 255).astype('uint8'))

        # Applying the Birds Eye Transformation
        #mask = mask_1ch
        mask_birdeye = BirdsEyePerspective(mask_1ch, capture_region)

        # Always pass a 1-D array to histogram function.
        # Hence, we need to convert from 3-channel color image to grayscale
        #processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

        left_detected = right_detected = False
        left_x = left_y = right_x = right_y = []

        # If there have been lanes detected in the past, the algorithm will first try to
        # find new lanes along the old one. This will improve performance

        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(mask_birdeye, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(mask_birdeye, self.right_line.best_fit_poly, self.line_segments)

            left_detected, right_detected = self.__check_lines(left_x, left_y, right_x, right_y)

        # If no lanes are found a histogram search will be performed
        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                mask_birdeye, self.line_segments, (self.image_offset, np.int(0.5*lane_masked_image.shape[1])), h_window=7)
            left_x, left_y = outlier_removal(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                mask_birdeye, self.line_segments, (np.int(0.5*lane_masked_image.shape[1]), lane_masked_image.shape[1] - self.image_offset), h_window=7)
            right_x, right_y = outlier_removal(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.__check_lines(left_x, left_y, right_x, right_y)

        print(left_detected, right_detected)

        # Updated left lane information.
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)

        # Add information onto the frame
        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = calc_curvature(self.center_poly)
            self.offset = (lane_masked_image.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700

            self.__draw_lane_overlay(orig_image, capture_region)
            self.__draw_info_panel(orig_image)


        return mask_birdeye, mask_1ch, lane_masked_image, roi_extracted_image, orig_image
