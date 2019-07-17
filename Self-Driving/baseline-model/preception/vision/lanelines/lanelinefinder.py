import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
%matplotlib inline
import math




def read_image(image_path):
    return mpimg.imread(image_path)

def read_image_and_print_dims(image_path):
    image = mpimg.imread(image_path)
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)
    return image

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    # define blank mask
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    # img is the output of houg_line: image with lines drawn on it
    # initial_img * α + img * β + λ
    return cv2.addWeighted(initial_img, α, img, β, λ)

def intersection_x(coef1, intercept1, coef2, intercept2):
    # return x-cord of intersection of two lines
    x = (intercept2-intercept1)/(coef1-coef2)
    return x

def draw_linear_regression_line(coef, intercept, intersection_x, img, imshape=[540, 960], color=[255, 0, 0], thickness=2)
    # Get starting and ending points of regression on line, ints
    print("Coef: ", coef, "Intercept: ", intercept, "intersection_x: ", intersection_x)
    point_one = (int(intersection_x), int(intersection_x * coef + intercept))

    if coef > 0:
        point_two = (imshape[1], int(imshape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))

    print("Point one: ", point_one, "Point two: ", point_two)

    # Draw line using cv2.line
    cv2.line(img, point_one, point_two, color, thickness)

def find_line_fit(slope_intercept):
    # slope_intercept: an array [[slope, intercept], [slope, intercept] ... ]

    # Init Array
    kept_slopes = []
    kept_intercepts = []
    print("Slope & Intercept: ", slope_intercept)

    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]

    # Remove points with slope not within 1.5 SD from mean
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

    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    print("Slope: ", slope, "Intercept: ", intercept)

    return slope, intercept


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # hardcoded image size
    imshape = [540, 960]

    # Init arrays
    positive_slope_points = []
    negative_slope_points = []
    positive_slope_intercept = []
    negative_slope_intercept = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1-y2) / (x1-x2)
            length = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )

            if not math.isnan(slope):
                # Add length requirement to determine if it is a lane
                if length > 50:
                    if slope > 0:
                        positive_slope_points.append([x1, y1])
                        positive_slope_points.append([x2, y2])

                        positive_slope_intercept.append([slope, y1-slope*x1])

                    elif slope < 0:
                        negative_slope_points.append([x1, y1])
                        negative_slope_points.append([x2, y2])

                        negative_slope_intercept.append([slope, y1-slope*x1])

    # For case where we currently both arrays (slope and intercept) are empty
    # Waive length requirement

    # 1 positive_slope_points array is empty
    if not positive_slope_points:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                if slope > 0:
                    positive_slope_points.append([x1, y1])
                    positive_slope_points.append([x2, y2])

                    positive_slope_intercept.append([slope, y1-slope*x1])
