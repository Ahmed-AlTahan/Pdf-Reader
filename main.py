from pdf2image import convert_from_path
from PageEditor import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
pdf2image_path = 'C:\\Users\\HP\\Desktop\\ComputerVisionLabs\\PdfReaderProject\\poppler-0.68.0\\poppler-0.68.0\\bin\\'

def get_pdf_as_images(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=pdf2image_path)
    result = []
    for i in range(len(images)):
        page = np.array(images[i],dtype=np.uint8)
        page = cv2.cvtColor(page,cv2.COLOR_BGR2RGB)
        result.append(page)
    return result

def Init_pdf(pdf_path , foregroundColor , backgroundColor , eyeComfortScale):
    print('pdf path in init::',pdf_path)
    pages_as_images = get_pdf_as_images(pdf_path=pdf_path)
    result = [blue_light_filter(image,value=eyeComfortScale/100.0) for image in pages_as_images]
    return result

p = 'pdfexample.pdf'
get_pdf_as_images(p)