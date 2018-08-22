import numpy as np
import math
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean
import numpy as np
import cv2


def mask_image(img, vertices):
    mask = np.zeros_like(img)
    # Deciding a 3-channel or 1-channel color to fill the mask depending on
    # whether the input image is in color or grayscale.
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_vertices(l,t,b,r,w,h):

    # Saints Row 3rd Person Cam vertices for bike
    # vertices = np.array([ [l,b], [l+np.floor(w/10),t+np.floor(2*h/10)], [r-np.floor(w/10), \
    # t+np.floor(2*h/10)], [r,b], \
    # [r-np.floor(w/3),b], [r-np.floor(w/2)+35, b-np.floor(w/2)],\
    #  [r-np.floor(w/2)-35, b-np.floor(w/2)], [r-np.floor(2*w/3), b] ], np.int32)

    # Vertices for 1st Person Cam NFS Run
    vertices = np.array([ [l,b], [l,t], [r,t], [r,b] ], np.int32)

    # return the chosen vertices
    return vertices

def find_lane_markers(img):

    #converted = convert_hls(img)
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0,   100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_img = cv2.bitwise_and(img, img, mask = mask)
    return masked_img



def find_line_fit(slope_intercept):
    """slope_intercept is an array [[slope, intercept], [slope, intercept]...]."""

    # Initialise arrays
    kept_slopes = []
    kept_intercepts = []
    #print("Slope & intercept: ", slope_intercept)
    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]

    # Remove points with slope not within 1.5 standard deviations of the mean
    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    slope_std = np.std(slopes)
    for pair in slope_intercept:
        slope = pair[0]
        if slope - mean_slope < 1.5 * slope_std:
            kept_slopes.append(slope)
            kept_intercepts.append(pair[1])
    if not kept_slopes:
        kept_slopes = slopes
        kept_intercepts = [pair[1] for pair in slope_intercept]
    # Take estimate of slope, intercept to be the mean of remaining values
    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept



def hough_lines(img, rho=1, theta=np.pi/180, threshold=15, min_line_length=50, max_line_gap=20):
    """
        'hough_lines' function applies HOUGHLINESP TRANSFORM to 'img' image

        <var_name> = <default_value>    # Description of variable
        rho             = 1             # distance resolution in pixels of the Hough grid
        theta           = np.pi / 180   # angular resolution in radians of the Hough grid
        threshold       = 15            # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50            # 50 minimum number of pixels making up a line
        max_line_gap    = 10            # 20 maximum gap in pixels between connectable line segments
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
                        minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def intersection_x(m1, c1, m2, c2):
    """
    Returns x-coordinate of intersection of two lines.
    Line1 --> y = m1*x + c1
    Line2 --> y = m2*x + c2
    At intersection point (xi,yi), yi = m1*xi + c1 = m2*xi + c2
    Therefore, xi = (c2-c1)/(m1-m2)
    """
    x = (c2-c1)/(m1-m2)
    return x

def draw_linear_regression_line(m, c, intersection_x, img, imshape=[480,640], color=[255, 0, 0], thickness=2):
    """
    Draws a straight line based on input slope m, intercept c
    Takes the starting point as input x-point intersection_x.
    Takes the end point with x as top-right or bot-right of the image
    depending on slope
    """

    # Get starting and ending points of regression line, ints.
    print("Slope Coef: ", m, "y Intercept: ", c,
          "intersection_x: ", intersection_x)
    point_one = (int(intersection_x), int(intersection_x * m + c))
    if m > 0:
        point_two = (imshape[1], int(imshape[1] * m + c))
    elif m < 0:
        point_two = (0, int(0 * m + c))
    print("Point one: ", point_one, "Point two: ", point_two)

    # Draw line using cv2.line
    cv2.line(img, point_one, point_two, color, thickness)



def draw_lines(img, lines, min_line_length=10, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Image parameters (hard-coded. TODO: Make not hard-coded.)
    imshape = [540, 960]

    # Initialise arrays
    positive_slope_points = []
    negative_slope_points = []
    positive_slope_intercept = []
    negative_slope_intercept = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            #print("Points: ", [x1, y1, x2, y2])
            length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            # print("Length: ", length)
            # if not(math.isnan(slope)):    NOT WORKING
            if (x1 != x2):
                if length > min_line_length:
                    if slope > 5:
                        positive_slope_points.append([x1, y1])
                        positive_slope_points.append([x2, y2])
                        positive_slope_intercept.append([slope, y1-slope*x1])
                    elif slope < 5:
                        negative_slope_points.append([x1, y1])
                        negative_slope_points.append([x2, y2])
                        negative_slope_intercept.append([slope, y1-slope*x1])

    # If either array is empty, waive length requirement
    # if not positive_slope_points:
    #     for line in lines:
    #         for x1,y1,x2,y2 in line:
    #             slope = (y1-y2)/(x1-x2)
    #             if np.bitwise_and(slope > 0, not(math.isnan(slope))):
    #                 positive_slope_points.append([x1, y1])
    #                 positive_slope_points.append([x2, y2])
    #                 positive_slope_intercept.append([slope, y1-slope*x1])
    # if not negative_slope_points:
    #     for line in lines:
    #         for x1,y1,x2,y2 in line:
    #             slope = (y1-y2)/(x1-x2)
    #             if np.bitwise_and(slope < 0, not(math.isnan(slope))):
    #                 negative_slope_points.append([x1, y1])
    #                 negative_slope_points.append([x2, y2])
    #                 negative_slope_intercept.append([slope, y1-slope*x1])
    if not positive_slope_points:
        print("positive_slope_points still empty")
    if not negative_slope_points:
        print("negative_slope_points still empty")
    # Even though positive_slope_points is not used, I am keeping it for debugging purposes.
    positive_slope_points = np.array(positive_slope_points)
    negative_slope_points = np.array(negative_slope_points)
    # print("Positive slope line points: ", positive_slope_points)
    # print("Negative slope line points: ", negative_slope_points)
    # print("positive slope points dtype: ", positive_slope_points.dtype)

    # Get intercept and coefficient of fitted lines
    print('Positive Slope Calcs')
    pos_coef, pos_intercept = find_line_fit(positive_slope_intercept)
    print('Negative Slope Calcs')
    neg_coef, neg_intercept = find_line_fit(negative_slope_intercept)

    # Discarded Linear Regression Option:
    # Get intercept and coefficient of linear regression lines
    # pos_coef, pos_intercept = find_linear_regression_line(positive_slope_points)
    # neg_coef, neg_intercept = find_linear_regression_line(negative_slope_points)

    # Get intersection point
    intersection_x_coord = intersection_x(pos_coef, pos_intercept, neg_coef, neg_intercept)

    # Plot lines
    draw_linear_regression_line(pos_coef, pos_intercept, intersection_x_coord, img)
    draw_linear_regression_line(neg_coef, neg_intercept, intersection_x_coord, img)


def draw_lanes(image, capture_region):
    """
    Draws Lanes on the input image
    """

    # topleftx, toplefty, botrightx, botrighty = capture_region
    tlx, tly, brx, bry = capture_region
    w = brx - tlx
    h = bry - tly
    t = tly     # t = tly = 40
    b = bry     # b = bry = 520
    l = tlx     # t = tlx = 0
    r = brx     # r = brx = 640

    imshape = image.shape

    # Convert Image to grayscale
    processed_image = find_lane_markers(image)
    image_gray = cv2.cvtColor(processed_image,cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    kernel_size = (5, 5)
    processed_image = cv2.GaussianBlur(image_gray, kernel_size, 0)
    # processed_image = cv2.adaptiveThreshold(processed_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,25, 15)

    # processed_image = cv2.Canny(processed_image,threshold1=50,threshold2=150)
    vertices = mask_vertices(l,t,b,r,w,h)
    processed_image = mask_image(processed_image, [vertices])

    # Applying HOUGHLINESP TRANSFORM
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # 50 minimum number of pixels making up a line
    max_line_gap = 10  # 20 maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    lines_image = hough_lines(processed_image, rho, theta, threshold, min_line_length, max_line_gap)

    # Convert Hough from single channel to RGB to prep for weighted
    # TODO: Have it convert the lines to red, not white.
    image_with_lanes_bgr = cv2.cvtColor(lines_image, cv2.COLOR_GRAY2BGR)
    # hough_rgb_image.dtype: uint8.  Shape: (540,960,3).
    # hough_rgb_image is like [[[0 0 0], [0 0 0],...] [[0 0 0], [0 0 0],...]]

    return processed_image, image_gray, image_with_lanes_bgr
