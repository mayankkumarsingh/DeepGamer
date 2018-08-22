import cv2
import numpy as np
from scipy import signal

src_bl = [0,310]
src_br = [640,310]
src_tl = [262,180]
src_tr = [360,180]

dest_bl = [272+30+0,480]
dest_br = [342+30-0,480]
dest_tl = [262,0]
dest_tr = [360,0]

# Initialziation for Histogram Plots
COLOR_BOX_BORDER = np.array([100, 160, 255],np.uint8)
COLOR_HIST_PLOT = np.array([225, 150, 50],np.uint8)
COLOR_POSITIVE_DETECTION = np.array([0, 150, 0],np.uint8)
COLOR_THRESHOLD = np.array([50, 225, 225],np.uint8)
COLOR_GRID = np.array([100, 150, 150],np.uint8)


def BirdsEyePerspective(img, capture_region):

    IMAGE_H = capture_region[3] - capture_region[1]
    IMAGE_W = capture_region[2] - capture_region[0]
    print(IMAGE_H,IMAGE_W)

    src = np.float32([src_tl, src_bl, src_br, src_tr])
    dst = np.float32([dest_tl, dest_bl, dest_br, dest_tr])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR) # Image warping

    return warped_img

def BirdsEyePerspectiveInverse(img, capture_region):

    IMAGE_H = capture_region[3] - capture_region[1]
    IMAGE_W = capture_region[2] - capture_region[0]
    print(IMAGE_H,IMAGE_W)

    src = np.float32([src_tl, src_bl, src_br, src_tr])
    dst = np.float32([dest_tl, dest_bl, dest_br, dest_tr])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    warped_img = cv2.warpPerspective(img, Minv, (IMAGE_W, IMAGE_H), flags=cv2.INTER_LINEAR) # Image warping

    return warped_img


def mask_image(img, vertices):
    mask = np.zeros_like(img)
    # Deciding a 3-channel or 1-channel color to fill the mask depending on
    # whether the input image is in color or grayscale.
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255.,) * channel_count
    else:
        ignore_mask_color = 255
    print(ignore_mask_color)
    cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_vertices(l,t,b,r,w,h):

    ## Saints Row 3rd Person Cam vertices for bike
    # vertices = np.array([ [l,b], [l+np.floor(w/10),t+np.floor(2*h/10)], [r-np.floor(w/10), \
    # t+np.floor(2*h/10)], [r,b], \
    # [r-np.floor(w/3),b], [r-np.floor(w/2)+35, b-np.floor(w/2)],\
    #  [r-np.floor(w/2)-35, b-np.floor(w/2)], [r-np.floor(2*w/3), b] ], np.int32)

    # Vertices for Car Dash  Cam NFS Run
    # vertices = np.array([ [l,t], [l,b-np.floor(h/3.5)], [l+np.floor(w/3),b-np.floor(h/3.175)], \
    #                 [r-np.floor(w/3),b-np.floor(h/3.175)], [r,b-np.floor(h/3.5)], [r,t] ], np.int32)

    # Vertices for Car Dash  Cam NFS Run + selecting ROI as only the road
    # For a 640 x 480 image, selection roi as (0,350), (640,350), (640,200), (450,170),(320,170), (0,200),(0,350)
    vertices = np.array([ [l,np.floor(0.707*h)], [r,np.floor(0.707*h)], [r,np.floor(0.545*h)], \
                [l+np.floor(0.54*w),np.floor(0.354*h)], [l+np.floor(0.3125*w),np.floor(0.354*h) ],\
                [l,np.floor(0.416*h)] ], np.int32)

    ## Vertices for 1st Person Cam in NFS Run
    #vertices = np.array([ [l,b], [l,t], [r,t], [r,b] ], np.int32)


    # return the chosen vertices
    return vertices


def abs_sobel(img_ch, orient='x', sobel_kernel=3):
    """
    Applies the sobel operation on a gray scale image.
    :param img_ch:
    :param orient: 'x' or 'y'
    :param sobel_kernel: an uneven integer
    :return:
    """
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    return abs_s

def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculates the magnitude of the gradient.
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)

def gradient_direction(sobel_x, sobel_y):
    """
    Calculates the direction of the gradient. NaN values cause by zero division will be replaced
    by the maximum value (np.pi / 2).
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)

def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return:
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def binary_noise_reduction(img, thresh):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


def extract_yellow(img):
    """
    Generates an image mask selecting yellow pixels.
    :param img: image with pixels in range 0-255
    :return: Yellow 255 not yellow 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 100), (65, 255, 255))

    return mask

def extract_dark(img):
    """
    Generates an image mask selecting dark pixels.
    :param img: image with pixels in range 0-255
    :return: Dark 255 not dark 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))
    return mask

def extract_white(img):
    """
    Generates an image mask selecting dark pixels.
    :param img: image with pixels in range 0-255
    :return: Dark 255 not dark 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = cv2.inRange(hsv, (0, 200, 0.), (255, 255, 255.))
    return mask

def extract_highlights(img, p=99.9):
    """
    Generates an image mask selecting highlights.
    :param p: percentile for highlight selection. default=99.9
    :param img: image with pixels in range 0-255
    :return: Highlight 255 not highlight 0
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask


def generate_lane_mask(img, v_cutoff=0):
    """
    Generates a binary mask selecting the lane lines of an street scene image.
    :param img: RGB color image
    :param v_cutoff: vertical cutoff to limit the search area
    :return: binary mask
    """
    window = img[v_cutoff:, :, :]
    # yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    # yuv = 255 - yuv
    # hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    # chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    # gray = np.mean(chs, 2)
    gray = cv2.cvtColor(window,cv2.COLOR_RGB2GRAY)
    #window = cv2.GaussianBlur(window,(5,5),1)

    canned = cv2.Canny(gray, threshold1=150, threshold2=300)
    # Output canned = single channel output 255 when edge detected, else 0
    #s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
    #s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

    #grad_dir = gradient_direction(s_x, s_y)
    #grad_mag = gradient_magnitude(s_x, s_y)

    ylw = extract_yellow(window)
    wht = extract_white(window)
    #highlights = extract_highlights(window[:, :, 0])

    mask_3ch = np.zeros(img.shape, dtype=np.uint8)

    # mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
    #                     (s_y >= 25) & (s_y <= 255)) |
    #                    ((grad_mag >= 30) & (grad_mag <= 512) &
    #                     (grad_dir >= 0.2) & (grad_dir <= 1.)) |
    #                    (ylw == 255) | (wht == 255) |
    #                    (highlights == 255)] = 1

    # FOR USE with cv2.Canny
    mask_3ch[v_cutoff:, :][(canned == 255) | (ylw == 255) | (wht == 255) ] = np.uint8(255)

    # FOR USE with sobel n gradient calcs
    # mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
    #                     (s_y >= 25) & (s_y <= 255)) |
    #                    ((grad_mag >= 30) & (grad_mag <= 512) &
    #                     (grad_dir >= 0.2) & (grad_dir <= 1.)) |
    #                     (ylw == 255) | (wht == 255) |
    #                    (highlights == 255)] = 255

    mask_3ch = binary_noise_reduction(mask_3ch, 4)
    # processed_image = cv2.bitwise_and(img, mask)
    return mask_3ch #processed_image

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


def histogram_lane_detection(img, steps, search_window, h_window, peak_threshold, frame_debug):

    """
    Tries to detect lane line pixels by applying a sliding histogram.
    :param img: binary image (Always pass a grayscale image to this fn)
    :param steps: steps for the sliding histogram
    :param search_window: Tuple which limits the horizontal search space.
    :param h_window: window size for horizontal histogram smoothing
    :param peak_threshold is the threshold for filtering peaks higher than this value
    :param frame_debug is a boolean flag that decides if we should generate histogram plot or not
    :return: x, y of detected pixels
    """
    all_x = {}
    all_y = {}
    ctr = 0
    masked_img = img[:, search_window[0]:search_window[1]]
    max_lane_width = 50
    pixels_per_step = img.shape[0] // steps
    print(steps)
    if frame_debug:
        histogram_overlay = np.zeros((img.shape[0],img.shape[1],3), np.uint8)

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        center = (start + end) // 2
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 30)))
        peaks = peaks[np.nonzero(histogram_smooth[peaks] > peak_threshold)]

        if(frame_debug):
            histogram_overlay[start-2:start,:,:] = COLOR_BOX_BORDER
            histogram_overlay[end-2:end,:,:] = COLOR_BOX_BORDER
            histogram_overlay[end:start,0:2,:] = COLOR_BOX_BORDER
            histogram_overlay[end:start,-2:,:] = COLOR_BOX_BORDER
            #histogram_overlay[end:start:(start-end)//4,1:-1:20,:] = COLOR_GRID
            hst_smooth_log = np.log1p(histogram_smooth)
            histogram_overlay[start-np.uint8(np.log1p(peak_threshold)*(pixels_per_step-4)/max(hst_smooth_log)),1:-1:1,:] = COLOR_THRESHOLD
            for peak in peaks:
                histogram_overlay[end:start:2,peak-1:peak+1,:] = COLOR_GRID
            hst = np.uint8(hst_smooth_log*(pixels_per_step-4)/max(hst_smooth_log))
            for i in range(len(histogram_smooth)-4):
                histogram_overlay[start-hst[i]-2:start-hst[i],i:i+2,:] = COLOR_HIST_PLOT
        print('Length of Peaks: ',len(peaks))
        if(len(peaks) > 0):
            for peak in peaks:
                existing_lane = False
                x,y = peak, center
                print('Current Peak: ', peak)
                if (len(all_x) == 0):
                    print('Creating First Entry in Dictionary')
                    all_x[ctr] = np.full((1,),x + search_window[0]) # create a numpy array with 1 entry x
                    all_y[ctr] = np.full((1,),y) # create a numpy array with 1 entry y
                    ctr = ctr + 1
                else:
                    for r in range(len(all_x)):
                        if(abs(x + search_window[0] - all_x[r][-1]) < max_lane_width):
                            print('Appending Point {},{} to Lane number {}'.format(x, y, r))
                            all_x[r] = np.append(all_x[r], x + search_window[0])
                            all_y[r] = np.append(all_y[r], y)
                            existing_lane = True
                            break
                    # Create a New entry if no corresponding lane exists
                    if (not existing_lane):
                        print('Creating a New lane - number ', ctr)
                        all_x[ctr] = np.full((1,),x + search_window[0]) # create a numpy array with 1 entry x
                        all_y[ctr] = np.full((1,),y) # create a numpy array with 1 entry y
                        ctr = ctr + 1

    print(len(all_x), len(all_y))
    for i in range(len(all_x)):
        all_x[i],all_y[i] = outlier_removal(all_x[i],all_y[i])

    if(frame_debug):
        print(histogram_overlay.shape)
        return histogram_overlay, all_x, all_y
    else:
        return all_x, all_y

# def fuse_peaks(peaks, fuse_dist = 15):
#     fused_peaks = []
#     for i in range(len(peaks)-1):
#         if abs(peaks[i] - peaks[i+1]) > fuse_dist:
#             fused_peaks.append(peaks[i])

def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    """
    Returns the n highest peaks of a histogram above a given threshold.
    :param histogram:
    :param peaks: list of peak indexes
    :param n: number of peaks to select     
    :param threshold:
    :return:
    """
    if len(peaks) == 0:
        return []

    peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []

    x, y = zip(*peak_list)
    x = list(x)

    if len(peak_list) < n:
        return x

    return x[:n]


def detect_lane_along_poly(img, poly, steps):
    """
    Slides a window along a polynomial an selects all pixels inside.
    :param img: binary image
    :param poly: polynomial to follow
    :param steps: number of steps for the sliding window
    :return: x, y as a list of detected pixels
    """
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        # choose start and end y coordinates to calculate center y coordinate
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        # poly is an input -- polynomial self.line.best_fit_poly
        # calculate x-coord corresponding to y-coord = center
        x = int(poly(center))

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y

def get_pixel_in_window(img, x_center, y_center, size):
    """
    returns selected pixel inside a window.
    :param img: binary image
    :param x_center: x coordinate of the window center
    :param y_center: y coordinate of the window center
    :param size: size of the window in pixel
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y

def calculate_lane_area(lanes, area_height, steps):
    """
    Returns a list of pixel coordinates marking the area between two lanes
    :param lanes: Tuple of Lines. Expects the line polynomials to be a function of y.
    :param area_height:
    :param steps:
    :return:
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = area_height // steps
        start = area_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)

def are_lanes_plausible(lane_one, lane_two, parallel_thresh=(0.0003, 0.55), dist_thresh=(300, 480)):
    """
    Checks if two lines are plausible lanes by comparing the curvature and distance.
    :param lane_one:
    :param lane_two:
    :param parallel_thresh: Tuple of float values representing the delta threshold for the
    first and second coefficient of the polynomials.
    :param dist_thresh: Tuple of integer values marking the lower and upper threshold
    for the distance between plausible lanes.
    :return:
    """
    is_parallel = lane_one.is_current_fit_parallel(lane_two, threshold=parallel_thresh)
    is_not_parallel = not(is_parallel)
    dist = lane_one.get_current_fit_distance(lane_two)
    print(dist)
    is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]
    print(is_not_parallel, is_plausible_dist)
    return is_parallel & is_plausible_dist

def draw_poly(img, poly, steps, color, thickness=5, dashed=False):
    """
    Draws a polynomial onto an image.
    :param img:
    :param poly:
    :param steps:
    :param color:
    :param thickness:
    :param dashed:
    :return:
    """
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img

def draw_poly_arr(img, poly, steps, color, thickness=5, dashed=False, tip_length=1):
    """
    Draws a polynomial onto an image using arrows.
    :param img:
    :param poly:
    :param steps:
    :param color:
    :param thickness:
    :param dashed:
    :param tip_length:
    :return:
    """
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.arrowedLine(img, end_point, start_point, color, thickness, tipLength=tip_length)

    return img

def outlier_removal(x, y, q=5):
    """
    Removes horizontal outliers based on a given percentile.
    :param x: x coordinates of pixels
    :param y: y coordinates of pixels
    :param q: percentile
    :return: cleaned coordinates (x, y)
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]

def check_lines(self, left_x, left_y, right_x, right_y):
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

    if line_plausible((left_x, left_y), (right_x, right_y)):
        left_detected = True
        right_detected = True
    elif left_line is not None and right_line is not None:
        if self.__line_plausible((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
            left_detected = True
        if self.__line_plausible((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
            right_detected = True

    return left_detected, right_detected
