import tkinter as tk
from tkinter.colorchooser import askcolor

import PIL
import PyPDF2
import cv2
import cv2 as cv
from tkinter import RIGHT, Y, END, BOTH, LEFT, NW, filedialog, HORIZONTAL
from PIL import ImageTk,Image
from pdf2image import convert_from_path
import numpy as np
from tkPDFViewer import tkPDFViewer as pdf
import GuiConnector
import os

from EyesTracker import EyesTracker

global X_Cor
global Y_Cor



class main_GUI :

    ### items ###
    window = None
    tf_pdf_path = None
    btn_browse_pdf = None
    btn_select_foreground_color = None
    btn_select_background_color = None
    eye_comfort_slider = None
    btn_apply = None
    btn_scroll_down = None
    btn_scroll_up = None
    label_image = None
    btn_mark_page = None
    btn_remove_mark = None
    MarkedPage = (None , None)
    btn_select_line = None
    btn_remove_line_mark = None
    MarkedLine = (None, None)
    btn_select_Line_color = None

    btn_open_pdf = None


    #### variables ####
    pdf_path = None
    foregroundColor = None
    backgroundColor = None
    textColor = None
    eyeComfortScale = None


    pdfReader = None
    images = None
    page_Number = None



    def __init__(self):

        self.window = tk.Tk()
        self.window.title("")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='black')
        self.tracker = EyesTracker()
        self.tf_pdf_path = tk.Text(self.window, height=2, width=50)
        self.btn_browse_pdf = tk.Button(self.window, text='Browse pdf file', command=self.BrowsePdf)
        self.btn_select_foreground_color = tk.Button(self.window, text='Select foreground color', command=self.SeclectForegroundColor)
        self.btn_select_background_color = tk.Button(self.window, text='Select background color', command=self.SeclectBackgroundColor)
        self.eye_comfort_slider = tk.Scale(self.window, from_=0, to=100, length=400 , tickinterval=10, orient=HORIZONTAL , command=self.SliderChanged)
        self.btn_apply = tk.Button(self.window, text='Apply',command=self.Apply)
        self.btn_scroll_down = tk.Button(self.window, text='Scroll down', command=self.ScrollDown)
        self.btn_scroll_up = tk.Button(self.window, text='Scroll up', command=self.ScrollUp)
        self.label_image = tk.Label()
        self.btn_mark_page = tk.Button(self.window, text='Mark page', command=self.MarkPage)
        self.btn_remove_mark = tk.Button(self.window, text='Remove Mark', command=self.remove_mark)

        self.btn_mark_page = tk.Button(self.window, text='Mark page', command=self.MarkPage)
        self.btn_remove_mark = tk.Button(self.window, text='Remove Mark', command=self.remove_mark)
        self.btn_select_line = tk.Button(self.window, text='Mark important text', command=self.MarkText)
        self.btn_remove_line_mark = tk.Button(self.window, text='Remove marked text', command=self.removeMarkedText)
        self.btn_select_Line_color = tk.Button(self.window, text='Select mark color ', command=self.SeclectMarkColor)


        # self.btn_open_pdf = tk.Button(self.window , text='Open', command=self.OpenPdf)




        self.SetPostions()

        self.page_Number = 0

        self.window.mainloop()

    def SetPostions(self) :

        self.tf_pdf_path.place(x=20, y=30)
        self.btn_browse_pdf.place(x=20, y=80)
        self.btn_select_foreground_color.place(x=20, y=150)
        self.btn_select_background_color.place(x=200, y=150)
        self.eye_comfort_slider.place(x = 20 , y = 200)
        self.btn_apply.place(x = 20 , y = 280)
        self.btn_scroll_up.place(x=700, y=300)
        self.btn_scroll_down.place(x=700, y=350)
        self.btn_mark_page.place(x=20, y=320)
        self.btn_remove_mark.place(x=120 , y=320)


        self.btn_select_Line_color.place(x=20 , y=380)
        self.btn_select_line.place(x=20 , y=420)
        #self.btn_remove_line_mark.place(x=150 , y=420)


        # self.btn_open_pdf.place(x=120, y=80)

    def BrowsePdf(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                              filetypes=(("Text files", "*.pdf*"), ("all files", "*.*")))

        self.tf_pdf_path.delete(1.0, END)
        self.tf_pdf_path.insert(END , filename)

        input = self.tf_pdf_path.get("1.0", END)
        input = input[:-1]
        path = r'{}'.format(input)
        self.pdf_path = path

        self.convert_pdf_to_images()
        self.show_pdf_as_image()

    def SeclectForegroundColor(self):
        color = askcolor(title="Tkinter Color Chooser")
        self.foregroundColor = color[0]

    def SeclectBackgroundColor(self):
        color = askcolor(title="Tkinter Color Chooser")
        self.backgroundColor = color[0]

    def SliderChanged(self , event):
        self.eyeComfortScale = self.eye_comfort_slider.get()

    def convert_from_opencv_image_to_pillow(self , image):
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        return img_pil

    def convert_from_pillow_image_to_opencv(self, image):
        opencv_img = np.asarray(image)
        opencv_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
        return opencv_img

    def convert_images_to_pdf(self):
        imagelist = self.images
        imagelist = [self.convert_from_pillow_image_to_opencv(image) for image in imagelist]
        Pil_imagelist = []
        for image in imagelist:
            Pil_imagelist.append(self.convert_from_opencv_image_to_pillow(image).convert('RGB'))

        self.pdf_path = r"C:\Users\Lenovo\Desktop\visionPfd.pdf"
        Pil_imagelist[0].save(self.pdf_path, save_all=True, append_images=Pil_imagelist[1:])

    def pdfViewer(self):

        self.convert_images_to_pdf()
        os.system(self.pdf_path)
        cv2.destroyAllWindows()
        self.tracker = EyesTracker()
        self.tracker.dun()

    def convert_pdf_to_images(self):
        pop_path = r"D:\university\vision\project\poppler-0.68.0\bin"
        images = convert_from_path(self.pdf_path , poppler_path=pop_path)
        self.images = images

    def Apply(self):

        imagelist = GuiConnector.init_pdf(self.pdf_path, self.foregroundColor, self.backgroundColor, self.eyeComfortScale)

        # 1 - convert images to Pil images
        # 2 - change images variable
        self.images = [self.convert_from_opencv_image_to_pillow(image) for image in imagelist]
        # 3 - show pdf as image
        self.show_pdf_as_image()
        self.pdfViewer()


    def show_pdf_as_image(self):
        self.label_image.place(x=850, y=20)
        img = self.images[self.page_Number]
        img = img.resize((630, 770))

        img = ImageTk.PhotoImage(img)
        self.label_image.configure(image=img)
        self.label_image.image = img

    def ScrollDown(self):
        if self.page_Number == len(self.images) - 1:
            return
        self.page_Number = self.page_Number + 1
        self.show_pdf_as_image()

    def ScrollUp(self):
        if self.page_Number == 0 :
            return

        self.page_Number = self.page_Number - 1
        self.show_pdf_as_image()

    def MarkPage(self):
        old_width = self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]).shape[1]
        old_hight = self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]).shape[0]
        new_width , new_hight = get_coordinates_of_mouse(self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]))
        global X_Cor
        global Y_Cor

        orignal_x = int(1.0 * (old_width * X_Cor) / new_width)
        orignal_y = int(1.0 * (old_hight * Y_Cor) / new_hight)

        self.remove_mark()

        image = self.images[self.page_Number]
        img = GuiConnector.marking_page(self.convert_from_pillow_image_to_opencv(image) , orignal_x , orignal_y)

        self.MarkedPage = (image , self.page_Number)
        self.images[self.page_Number] = self.convert_from_opencv_image_to_pillow(img)

        self.show_pdf_as_image()
        self.pdfViewer()

    def remove_mark(self):
        if self.MarkedPage[1] is not None :
            self.images[self.MarkedPage[1]] = self.MarkedPage[0]

        self.show_pdf_as_image()
        self.pdfViewer()

    def MarkText(self):
        old_width = self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]).shape[1]
        old_hight = self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]).shape[0]

        new_width, new_hight = get_coordinates_of_mouse(
            self.convert_from_pillow_image_to_opencv(self.images[self.page_Number]))
        global X_Cor
        global Y_Cor

        orignal_x = int(1.0 * (old_width * X_Cor) / new_width)
        orignal_y = int(1.0 * (old_hight * Y_Cor) / new_hight)


        image = self.images[self.page_Number]
        img = GuiConnector.marking_text(self.convert_from_pillow_image_to_opencv(image), orignal_x, orignal_y , self.textColor)

        self.MarkedLine = (image, self.page_Number)
        self.images[self.page_Number] = self.convert_from_opencv_image_to_pillow(img)

        self.show_pdf_as_image()
        self.pdfViewer()

    def removeMarkedText(self):
        if self.MarkedLine[1] is not None:
            self.images[self.MarkedLine[1]] = self.MarkedLine[0]

        self.show_pdf_as_image()
        self.pdfViewer()

    def SeclectMarkColor(self):
        color = askcolor(title="Tkinter Color Chooser")
        self.textColor = color[0]




def click_event(event , x , y ,flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
             global X_Cor
             global Y_Cor
             X_Cor = x
             Y_Cor = y

def get_coordinates_of_mouse(image):
    width = 750
    hight = 850
    image = cv.resize(image , (width , hight))
    cv.imshow('image', image)
    cv.setMouseCallback('image', click_event)
    cv.waitKey()
    return width , hight



def start() :

    gui = main_GUI()

    pass

start()