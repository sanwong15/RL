import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
from sklearn.linear_model import LinearRegression




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

def draw_linear_regression_line(coef, intercept, intersection_x, img, imshape=[540, 960], color=[255, 0, 0], thickness=2):
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
    # return a line representation (slope, intercept) to summarize a group of lines

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

    # STEP 1: We set a length requirement = 50
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

    # STEP 2: if none of the line detected actually pass the length requirement => we waive this requirement
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

    # 2 negative_slope_points array is empty
    if not negative_slope_points:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                if slope < 0:
                    negative_slope_points.append([x1, y1])
                    negative_slope_points.append([x2, y2])

                    negative_slope_intercept.append([slope, y1-slope*x1])


    # STEP 3: What if after we waive the length requirement and still got nothing -> we just tell the driver we can't find line
    if not positive_slope_points:
        print("positive_slope_points still empty")

    if not negative_slope_points:
        print("negative_slope_points still empty")

    #positive_slope_points = np.array(positive_slope_points)
    #negative_slope_points = np.array(negative_slope_points)


    # Find/Summarize Line candidate and represent in Coef and Intercept format
    pos_coef, pos_intercept = find_line_fit(positive_slope_intercept)
    neg_coef, neg_intercept = find_line_fit(negative_slope_intercept)

    # Get intersection point in order to draw a line
    intersection_x_coord = intersection_x(pos_coef, pos_intercept, neg_coef, neg_intercept)

    # Draw Line on image
    draw_linear_regression_line(pos_coef, pos_intercept, intersection_x_coord, img)
    draw_linear_regression_line(neg_coef, neg_intercept, intersection_x_coord, img)


# NOTE: No use at this point
def find_linear_regression_line(points):
    # find a linear regression line from points

    # Separate points into X and Y to fit LinearRegression model
    points_x = [[point[0]] for point in points]
    points_y = [point[1] for point in points]

    # Fit points to LinearRegression line
    clf = LinearRegression().fit(points_x, points_y)

    # Get parameters from line
    coef = clf.coef_[0]
    intercept = clf.intercept_
    print("Coefficients: ", coef, "Intercept: ", intercept)

    return coef, intercept


# MAIN FUNCTION TO BE CALLED (Pipeline function)
def draw_lane_lines(image):
    # Input: original image (collect from front facing car camera)
    # Output: Detect lane overlay on the original image

    imshape = image.shape

    # Step 1: Greyscale
    greyscaled_image = grayscale(image)

    # Step 2: Gaussian Blur
    blurred_grey_image = gaussian_blur(greyscaled_image, 5)

    # Step 3: Canny edge detection
    edges_image = canny(blurred_grey_image, 50, 150)

    # Step 4: Mask edges image
    border = 0
    vertices = np.array([[(0, imshape[0]), (465, 320), (475, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    edges_image_with_mask = region_of_interest(edges_image, vertices)
    bw_edges_image_with_mask = cv2.cvtColor(edges_image_with_mask, cv2.COLOR_GRAY2BGR)

    # Step 5: Hough Lines detection
    rho = 2
    theta = np.pi/180
    threshold = 45
    min_line_len = 40
    max_line_gap = 100
    lines_image = hough_lines(edges_image_with_mask, rho, theta, threshold, min_line_len, max_line_gap)

    # Step 6: Convert Hough from single channel to RGB to prepare for weighted
    hough_rgb_image = cv2.cvtColor(lines_image, cv2.COLOR_GRAY2BGR)

    # Step 7: Combine lines image with original image
    final_image = weighted_img(hough_rgb_image, image)

    return final_image





