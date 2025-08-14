import matplotlib.pyplot as  plt
import numpy as np
import cv2


def grid_plots(images: list, titles: list, figsize: tuple = (15, 15), shape: tuple = None, cmap: str = None):
    # convert images to nd array
    images_array = np.empty(len(images), dtype=object)
    for i in range(len(images)):
        images_array[i] = images[i]
    # get the shape of this array
    rows, cols = (images_array.shape[0], 1) if shape is None else shape[:2]
    # create the figure depending on images array shape
    fig, axis = plt.subplots(rows, cols, figsize=figsize)
    # flatting the axis array from (N,M) shape to (N*M,) shape
    axis = axis.ravel()
    # passing on all images array and plot the image
    for i in range(rows):
        for j in range(cols):
            img = images[i * cols + j]
            axis[i * cols + j].imshow(img, cmap)
            axis[i * cols + j].axis('off')
            if len(titles) > 0 and titles[i * cols + j] is not None:
                axis[i * cols + j].set_title(titles[i * cols + j])
            # don't show number on the axis
            axis[i * cols + j].axes.yaxis.set_visible(False)
            axis[i * cols + j].axes.xaxis.set_visible(False)
    plt.show()


def correcting_rgb_brightness(image: np.ndarray, brightness_threshold: int = 100, equalization: bool = False):
    # convert input image from RGB to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if equalization is True:
        gray = cv2.equalizeHist(gray)
    # bright pixel : is pixel has gray scale value larger the brightness_threshold
    #  dark pixel : is pixel has gray scale value less or equal to the brightness_threshold
    # get number of bright pixels
    num_of_bright_pixels = gray[gray > brightness_threshold].shape[0]
    # get number of dark pixels
    num_of_dark_pixels = gray[gray <= brightness_threshold].shape[0]
    # if number of dark pixels larger than bright pixels return inverse bgr image
    if num_of_dark_pixels > num_of_bright_pixels:
        return 255 - image, True
    # else return original image
    return image, False

def shrink_image(image: np.ndarray, shrinking_factor: int):
    # image shrinking depending on shrinking factor
    # get the new height and width of the image
    h = int(image.shape[0] / shrinking_factor)
    w = int(image.shape[1] / shrinking_factor)
    # shrink image using openCv resize image function
    shrinked_image = cv2.resize(image, (w, h))
    # return the shrinked image
    return shrinked_image

def get_number_of_boxes(image: np.ndarray):
    # find external contours
    cnts = cv2.findContours(255 - image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = list(cnts[0]) if len(cnts) == 2 else cnts[1]
    number_of_cnts = len(cnts)
    return number_of_cnts ,cnts

def cvt_to_gray(image: np.ndarray, enhancing: bool = False, structuring_element_kernel_size: tuple = (5, 5)):
    out_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if enhancing is True:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, structuring_element_kernel_size)
        bg = cv2.morphologyEx(out_gray, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(out_gray, bg, scale=255)
    return out_gray
def cvt_to_binary(image: np.ndarray, adaptive_method: int = -1, threshold_type: int = cv2.THRESH_BINARY,
                  threshold_block_size: int = 15, C: int = 8,
                  openning: int = False):
    if adaptive_method is None:
        out_binary = cv2.threshold(image, 127, 255, threshold_type)[1]
    elif adaptive_method in [cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE]:
        out_binary = cv2.threshold(image, 0, 255, adaptive_method + threshold_type)[1]
    else:
        out_binary = cv2.adaptiveThreshold(image, 255, adaptive_method, threshold_type, threshold_block_size, C)
    if openning is True:
        kernel = np.ones((3, 3), np.uint8)
        out_binary = 255 - cv2.morphologyEx(255 - out_binary, cv2.MORPH_OPEN, kernel)
    return out_binary

def draw_bound(bound: list, image: np.ndarray, bound_color: tuple = (255, 255, 255), bound_thickness: int = 1):
    p1 = (int(bound[0]), int(bound[1]))
    p2 = (int(bound[0] + bound[2]), int(bound[1] + bound[3]))
    cv2.rectangle(image, p1, p2, bound_color, bound_thickness, 1)


def crop_image(bound: tuple, image: np.ndarray):
    p1 = (int(bound[0]), int(bound[1]))
    p2 = (int(bound[0] + bound[2]), int(bound[1] + bound[3]))
    ROI = image[p1[1]:p2[1], p1[0]: p2[0]]
    return ROI.copy()


def get_biggest_contour(edges: np.ndarray, edges_thickness: int = -1, th_biggest: int = 0):
    # apply cv2.findContours algorithm on the image (edges) for only external contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros(edges.shape, np.uint8)
    if len(contours) == 0:
        return external_contours, None
    contours = list(contours)
    contours.sort(reverse=True, key=cv2.contourArea)
    # create a pure black image has the same size of input image (edges)
    # draw the biggest contour on the above image
    #cv2.drawContours(external_contours, contours, th_biggest, 255, edges_thickness)
    bound = cv2.boundingRect(contours[th_biggest])
    bound = list(bound)
    #bound[2] = 50
    #bound[3] = 50
    #bound[1] = 125
    screenshot = (bound[2]/bound[3] > 2) and bound[2] > 0.5 * edges.shape[1]
    x = int((2*bound[0] + bound[2]) / 2)
    y = 125
    p1 = (int(bound[0]), int(bound[1]))
    p2 = (int(bound[0] + bound[2]), int(bound[1] + bound[3]))
    cv2.circle(external_contours,(x,y),5,255,-1,1)
    #cv2.rectangle(external_contours, p1, p2, 255, -1, 1)
    return external_contours, contours[th_biggest], x ,screenshot

def get_bounded_contours_image(image_shape ,cnts):
    bounded_contours_image = np.zeros(image_shape ,np.uint8)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        bounded_contours_image = cv2.rectangle(bounded_contours_image ,(x,y),(x+w,y+h),255,thickness=-1)
    return bounded_contours_image