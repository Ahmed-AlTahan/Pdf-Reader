import numpy as np
from pdf2image import convert_from_path

import GuiConnector
import PageEditor
import cv2 as cv

pdf2image_path = r"D:\university\vision\project\poppler-0.68.0\bin"


#################################################################################################################
def init_pdf(pdf_path, foreground_color, background_color, eye_comfort_scale):
    if eye_comfort_scale:
        eye_comfort_scale = float(1.0 * eye_comfort_scale / 100)

    if background_color is not None:
        background_color = background_color[::-1]

    if foreground_color is not None:
        foreground_color = foreground_color[::-1]

    images = convert_pdf_to_images(pdf_path)
    images = [convert_from_pillow_image_to_opencv(image) for image in images]
    images = [PageEditor.change_page_colors(image, foreground_color, background_color) for image in images]
    images = [PageEditor.blue_light_filter(image, 'scale', eye_comfort_scale) for image in images]

    return images


#################################################################################################################

def convert_pdf_to_images(pdf_path):
    """
    :param pdf_path: the PDF file full path
    :return: images: the PDF file as a list of RGB images
    """
    images = convert_from_path(pdf_path, poppler_path=GuiConnector.pdf2image_path)
    return images


#################################################################################################################
def convert_from_pillow_image_to_opencv(image):
    """
    :param image: convert a PIL image (RGB) to OpenCV image (BGR)
    :return:
    """
    opencv_img = np.asarray(image)
    opencv_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
    return opencv_img

##################################################################################################################
def marking_text(image , x , y , color) :
    return PageEditor.highlight_row(image,x,y,color)

#################################################################################################################
def marking_page(image, x, y):
    return PageEditor.mark_page(image, x, y)
