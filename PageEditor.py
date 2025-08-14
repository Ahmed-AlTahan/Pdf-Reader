import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import ImageUtilities as iu

#################################################################################################################
"""
[1] Eye Comfort (Blue light filter)
image is an numpy array RGB so i'll remove a amount from the blue channel
this amount is Fixed value or Scale from the original blue value image or Log of the original blue value
the amount computed for each pixel for Log and Scale method depending on his blue value
resource : https://www.viewsonic.com/library/business/blue-light-filter-eye-strain/
"""


def blue_light_filter(image: np.ndarray, method: str = 'scale', value: float = 0.3):
    """
    :param image: colored input image as nd array in RGB order
    :param method: filtering method, default = 'scale'
    :param value: value of the filtering method, default = 0.3
    :return: filtered_image: the blue light filtered image
    """
    if value is None:
        return image
    # convert image to float
    filtered_image = image.astype(np.float64)
    # fixed method remove fixed value from blue pixels
    # Bnew = Bold - Constant
    if method == 'fixed':
        filtered_image[:, :, 0] -= value
    # scale method remove scale from blue pixels
    # Bnew = Bold - Constant * Bold
    elif method == 'scale':
        filtered_image[:, :, 0] -= value * filtered_image[:, :, 0]
    # log method remove log scale from blue pixels
    # Bnew = Bold - Constant * Log(Bold)
    elif method == 'log':
        filtered_image[:, :, 0] -= value * np.log(0.00001 + filtered_image[:, :, 0])
    filtered_image[image[:, :, 0] < 0] = 0
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image


#################################################################################################################
"""
[2] & [3] Change paper and font Colors
we will convert each pixel from the page and font pixel to a custom color
using this pipeline:
1. apply bilateral filter on the image to reduce the noise with respect of the edge
2. correct this RGB image brightness so i have a bright back ground and dark foreground
3. get a image in gray scale then thresholding it using height threshold
4. compute the big contours mask by extract all big contours in the threshold image
5. apply k-means clustering on the image pixels
6. most cluster samples is background cluster
7. second most cluster samples is foreground cluster
8. make their cluster centers as user need
8. using bitwise and operation for the big contour mask with original image  
9. using bitwise and operation for the inverse big contour mask with original image  
10. combine 8 and 9 and return the result
the easiest and fastest way is by applying thresholding on the image and 
each white pixel in the threshold image is a background pixel in the original image 
and each black pixel in the threshold image foreground pixel in the original image 
the second is by using CMY color space
"""


def change_page_colors(image: np.ndarray, fg_color: tuple = None,
                       bg_color: tuple = None):
    """
    :param image: colored input image as nd array in RGB order
    :param fg_color: the font color i want convert to
    :param bg_color: the page color i want convert to
    :return: converted_image : the result
    """
    # if both color are none the there are no need to compute anything so return the same input image
    if fg_color is None and bg_color is None:
        return image
    # apply bilateral filter on the image to reduce noise with respect the edges with kernel size = 1% * image_width
    #k_size = int(image.shape[1] / 100)
    #image = cv2.bilateralFilter(image, k_size, 20, 75)
    # get copy of the image
    original = image.copy()
    image=image.copy()
    # correct RGB Brightness
    image, corrected = iu.correcting_rgb_brightness(image)
    # convert the image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # get the gray image threshold
    binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]
    # compute the big contours mask
    big_contours_mask = get_big_contours_mask(255 - binary, threshold=10000)
    # get inverse of this big contours mask which represent the normal image contours (only words)
    normal_mask = 255 - big_contours_mask
    # extract the big contours RGB image using bitwise and operator between mask and original image
    big_contours = cv2.bitwise_and(big_contours_mask, original)
    # shrink the image 100 times to improve k-means training speed
    new_size = (int(image.shape[1] / 10), int(image.shape[0] / 10))
    image[big_contours_mask == 255] = 255
    original[big_contours_mask == 255] = 0
    resized_image = cv2.resize(image, new_size)
    # reshape the image to readable form for the k-means algorithm
    x = resized_image.reshape(-1, 3)
    # get a good number of clusters
    # apply k-means clustering on the image pixels where each sample is a one pixel has three features R,G,B
    kmeans = elbow_method(x)
    # get the clusters centers and convert him to uint8
    center = np.uint8(kmeans.cluster_centers_)
    # predict the original image using the trained k-means model after reshape it
    # label array is array contains the index of each pixel in center array
    x = original.reshape(-1, 3)
    label = kmeans.predict(x)
    # get number of iteration of each label
    unique, counts = np.unique(label.flatten(), return_counts=True)
    # sort above array depending on label counts (iterations)
    label_counts = np.asarray((unique, counts)).T
    sorted_array = label_counts[np.argsort(-label_counts[:, 1])]
    # the most common label is background pixel so convert his center to a bg_color
    if bg_color:
        background_label = sorted_array[0][0]
        center[background_label] = bg_color
    if fg_color:
        # the 2'th common label is foreground pixel so convert his center to a bg_color
        foreground_label = sorted_array[1][0]
        center[foreground_label] = fg_color
        # if the 3'th common label counts is larger than 2'th common label then his center is also a foreground pixel
        if sorted_array[2][1] / sorted_array[1][1] < 0.7:
            second_foreground_label = sorted_array[2][0]
            center[second_foreground_label] = fg_color
    # assign each pixel to his center so the font of n
    result = center[label.flatten()]
    # reshape the result to original image shape
    result = result.reshape(original.shape)
    # get only not big contours between converted image and the normal mask
    result = cv2.bitwise_and(normal_mask, result)
    # combine the above image with the biggest contours image and this is the final result
    converted_image = result + big_contours
    # return the result
    return converted_image.astype(np.uint8)


def elbow_method(data, maximum_number_clusters=10):
    # this method return a fitted K-means with a good number of clusters using elbow method
    wcss = []
    kmeans = None
    for i in range(1, maximum_number_clusters):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        if i > 2 and wcss[-2] / wcss[-1] < 1.3:
            print(i)
            return kmeans
    return kmeans


def get_big_contours_mask(image, threshold=None):
    # image is a binary image
    if threshold is None:
        threshold = 0.003 * (image.shape[0] * image.shape[1])
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    if len(contours) == 0:
        return external_contours, None
    for c in list(contours):
        bound = list(cv2.boundingRect(c))
        if (bound[2] * bound[3] > threshold) and (bound[2] * bound[3] < 0.8 * image.shape[0] * image.shape[1]) or (bound[2] > 0.6*image.shape[1] or bound[3] > 0.6*image.shape[0]):
            external_contours = cv2.drawContours(external_contours, [c], 0, (255, 255, 255), -1, 1)
    return external_contours


#################################################################################################################
"""
[4] Mark Page
this method mark a page by draw a circle in a place on a page
"""


def mark_page(image, x, y):
    radius = 50
    color = (255, 0, 0)
    thickness = 4
    image = cv2.circle(image, (x, y), radius, color, thickness)
    return image


#################################################################################################################
"""
Bonus [1]: Deployment as a Mobile App
    NOT IMPLEMENTED
"""

#################################################################################################################
"""
Bonus [2]: Highlight Text
"""


class DilationBasedTextDetector:
    def __init__(self):
        self.supported_adaptive_methods = [cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, -1]
        self.supported_threshold_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TOZERO,
                                          cv2.THRESH_TOZERO_INV, cv2.THRESH_TRUNC]

    def remove_outliers(self, image: np.ndarray, ignore_threshold: int = 100, removal_threshold: int = 15):
        cnts = cv2.findContours(255 - image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(cnts[0]) if len(cnts) == 2 else cnts[1]
        widths = []
        heights = []
        areas = []
        valid_cnts = []
        cnts.sort(reverse=True, key=cv2.contourArea)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if ignore_threshold < w * h < 250 * ignore_threshold and w < int(0.4 * image.shape[1]) and h < int(
                    0.4 * image.shape[0]):
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                valid_cnts.append(c)
        areas = np.array(areas)
        heights = np.array(heights)
        widths = np.array(widths)
        areas_mean = np.mean(areas)
        areas_std = np.std(areas)
        heights_mean = np.mean(heights)
        heights_std = np.std(heights)
        widths_mean = np.mean(widths)
        widths_std = np.std(widths)
        binary_without_outliers = np.zeros(image.shape, dtype=np.uint8)
        for area, height, width, c in zip(areas, heights, widths, valid_cnts):
            z_area = (area - areas_mean) / areas_std
            z_height = (height - heights_mean) / heights_std
            z_width = (width - widths_mean) / widths_std
            if z_area < removal_threshold and z_height < removal_threshold and z_width < removal_threshold:
                cv2.drawContours(binary_without_outliers, [c], 0, (255, 255, 255), -1)
        binary_without_outliers = 255 - binary_without_outliers
        number_of_black_pixels = binary_without_outliers[binary_without_outliers == 0].shape[0]
        return binary_without_outliers

    def get_boxes_image(self, image: np.ndarray, kernal_shape: tuple, number_of_boxes: int = 6, dead_line: int = 50):
        kernel = np.ones(kernal_shape, np.uint8)
        boxes_image = 255 - image
        iterations = 1
        while True:
            if iterations >= dead_line:
                break
            curr_number_of_boxes = iu.get_number_of_boxes(255 - boxes_image)[0]
            if curr_number_of_boxes <= number_of_boxes:
                return 255 - boxes_image
            boxes_image = cv2.dilate(boxes_image, kernel)
            iterations += 1
        return -1

    def detect_text(self, image: np.ndarray, number_of_boxes: int = 6, src_resize: tuple = (850, 1100)):
        image = cv2.resize(image, src_resize).copy()
        blurring_size = max(2 * (int(0.001 * max(image.shape[0], image.shape[1]))) + 1, 1)
        image = iu.correcting_rgb_brightness(image, 127)[0]
        image = cv2.bilateralFilter(image, blurring_size, 75, 75)
        structural_kernel_size = max(2 * (int(0.012 * max(image.shape[0], image.shape[1]))) + 1, 1)
        gray = iu.cvt_to_gray(image, True, (structural_kernel_size, structural_kernel_size))
        binary = iu.cvt_to_binary(gray, adaptive_method=cv2.THRESH_OTSU, threshold_type=cv2.THRESH_BINARY,
                                  openning=False)
        binary_without_outliers = self.remove_outliers(binary.copy(), ignore_threshold=50, removal_threshold=6)
        dilation_kernel_shape = (max(int(0.001 * image.shape[0]), 2), max(int(0.003 * image.shape[1]), 2))
        boxes_image = self.get_boxes_image(binary_without_outliers, dilation_kernel_shape, number_of_boxes)
        return boxes_image
def check_action(event, x, y, flags, param):
    global image
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('clicked')
        image = highlight_row(image,x, y)


def create_blank(shape, rgb_color=(0, 0, 0)):
    # output is bgr image
    image = np.zeros((shape[0], shape[1], 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image

def highlight_row(image,x,y,highlight_color = (0,255,0)):
    print(image.shape)
    image_size = (image.shape[1],image.shape[0])
    text_detector = DilationBasedTextDetector()
    text_contours_image = text_detector.detect_text(image, number_of_boxes=20, src_resize=(1700,2200))
    number_of_text_contours, text_contours = iu.get_number_of_boxes(text_contours_image)
    bounded_text_contours_image = iu.get_bounded_contours_image((2200,1700), text_contours)
    _, rows_contours = iu.get_number_of_boxes(255 - bounded_text_contours_image)
    for cnt in rows_contours:
        if cv2.pointPolygonTest(cnt, (x, y), False) == 1:
            clicked_area_mask = iu.get_bounded_contours_image(bounded_text_contours_image.shape, [cnt])
            clicked_area_mask = cv2.merge((clicked_area_mask.copy(), clicked_area_mask.copy(), clicked_area_mask.copy()))
            inv_mask = 255 - clicked_area_mask
            highlight_color_blank_image = create_blank(clicked_area_mask.shape, highlight_color)
            b = (cv2.bitwise_and(clicked_area_mask, highlight_color_blank_image)).astype(np.uint8)
            r = cv2.addWeighted(b, 0.1, image, 0.9, 0)
            r = cv2.bitwise_and(r, clicked_area_mask)
            f = (cv2.bitwise_and(inv_mask, image)).astype(np.uint8)
            return r + f
    return image


# image = cv2.imread('page0.jpg')
# #image = cv2.resize(image, (int(image.shape[1] / 5), int(image.shape[0] / 5))).copy()
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',check_action)
# print(image.shape)
# while (1):
#     cv2.imshow('image', image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()
# plt.show()
