# import time
#
# import cv2
# import sys
# import numpy as np
# import pyautogui as pyautogui
#
# import ImageUtilities
# import TrackerUtilits
# from eye_classes import decision
#
#
# class EyesTracker:
#     def __init__(self, tracker_type='MIL'):
#         self.tracker_type = tracker_type
#         self.face_cascade = cv2.CascadeClassifier('Files\\haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier('Files\\haarcascade_eye.xml')
#         self.tracker = TrackerUtilits.create_tracker(self.tracker_type)
#         self.video_stream = cv2.VideoCapture(0)
#         self.last_detect_time = 0
#         if not self.video_stream.isOpened():
#             print("Could not open video")
#             sys.exit()
#         ok, frame = self.video_stream.read()
#         if not ok:
#             print('Cannot read video file')
#             sys.exit()
#         self.curr_frame = frame
#         self.origin = self.curr_frame.copy()
#         self.ROI = None
#         self.area_to_search = None
#         self.frame_count = 0
#         self.last_x = 0
#         self.screenshot_counter = 0
#         self.change_acc = 1
#         self.max_change = 1
#         cv2.namedWindow('Tracking')
#         cv2.createTrackbar('threshold', 'Tracking', 25, 255, self.do_nothing)
#
#     def do_nothing(self):
#         pass
#
#     def detect_faces(self, img):
#         frame = None
#         gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_frame = cv2.equalizeHist(gray_frame)
#         cv2.imshow('gray frame', gray_frame)
#         coords = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
#         if len(coords) > 1:
#             biggest = (0, 0, 0, 0)
#             for i in coords:
#                 if i[3] * i[2] > biggest[3] * biggest[2]:
#                     biggest = i
#             biggest = np.array([i], np.int32)
#         elif len(coords) == 1:
#             biggest = coords
#         else:
#             return None
#         for (x, y, w, h) in biggest:
#             frame = img[y:y + h, x:x + w]
#         return (frame, biggest[-1])
#
#     def detect_eyes(self, img, point, erode=10):
#         gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         #gray_frame = cv2.equalizeHist(gray_frame)
#         eyes = self.eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
#         width = np.size(img, 1)  # get face frame width
#         height = np.size(img, 0)  # get face frame height
#         left_eye, right_eye = None, None
#         for (x, y, w, h) in eyes:
#             if y > height / 2:
#                 pass
#             eyecenter = x + w / 2  # get the eye center
#             if eyecenter < width * 0.5:
#                 left_eye = ((point[0] + x, point[1] + y + erode, w, h - erode), img[y + erode:y + h - erode, x:x + w])
#                 return left_eye
#             else:
#                 right_eye = ((point[0] + x, point[1] + y + erode, w, h - erode), img[y + erode:y + h - erode, x:x + w])
#                 return right_eye
#         return None
#
#     def try_extract_features(self, frame):
#         if frame is None:
#             return None
#         face_detection_results = self.detect_faces(frame)
#         if face_detection_results is None:
#             return None
#         face_frame = face_detection_results[0]
#         eyes_detection_result = self.detect_eyes(face_frame, face_detection_results[1])
#         if eyes_detection_result is not None:
#             return eyes_detection_result
#         return None
#
#     def extract_features(self):
#         i = 0
#         while True:
#             eyes_detection_result = self.try_extract_features(frame=self.origin)
#             if eyes_detection_result is None and i > 15:
#                 print('Can\'t detect a face please check about lighting condition and try again....')
#                 raise Exception
#             elif eyes_detection_result is not None and i <= 15:
#                 self.last_detect_time = time.time()
#                 return eyes_detection_result
#             else:
#                 self.read_frame()
#                 i += 1
#
#     def read_frame(self):
#         ok, frame = self.video_stream.read()
#         if not ok:
#             print('Can\'t read from the video stream')
#             sys.exit(-1)
#         self.curr_frame = cv2.flip(frame, 1)
#         self.origin = self.curr_frame.copy()
#
#     def process(self):
#         # self.decisor.start(eye = self.ROI ,threshold = cv2.getTrackbarPos('threshold', 'Tracking'))
#         roi = cv2.resize(self.ROI, (300, 300))
#         #roi_without = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
#         #roi_without[:, :, 0] = np.uint8(0.5 * roi_without[:, :, 0])
#         #roi_without = cv2.cvtColor(roi_without, cv2.COLOR_YUV2BGR)
#         #cv2.imshow('ROI without light', roi_without)
#         cv2.imshow('ROI', roi)
#         r = roi.copy()
#         r = cv2.bilateralFilter(r, 11, 50, 25)
#         r = ImageUtilities.cvt_to_gray(r, enhancing=False)
#         r = cv2.equalizeHist(r)
#         cv2.imshow('gray', r)
#         threshold = cv2.getTrackbarPos('threshold', 'Tracking')
#         r = cv2.threshold(r, threshold, 255, cv2.THRESH_BINARY_INV)[1]
#         # r = cv2.adaptiveThreshold(r ,255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,max(2*threshold+1,3),8)
#         r = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel=np.ones((5, 5), dtype=np.uint8), iterations=4)
#         cv2.imshow('filtered roi', r)
#         #r = cv2.erode(r ,np.ones((3,10),dtype=np.uint8),iterations=5)
#
#         r, _, x ,is_screen = ImageUtilities.get_biggest_contour(r)
#         if is_screen:
#             if self.screenshot_counter > 15:
#                 print('Screen shot')
#                 im1 = pyautogui.screenshot()
#                 im2 = pyautogui.screenshot('my_screenshot.png')
#                 self.change_acc = 0
#                 self.last_x = 150
#                 self.screenshot_counter = 0
#             else :
#                 self.screenshot_counter += 1
#         elif self.frame_count % 30 == 0:
#             #print('stop')
#             #print('acc = ',np.int64(self.change_acc))
#             #self.max_change = max(np.int64(self.max_change),np.int64(self.change_acc))
#             #print('max = ',self.max_change)
#             thr = 10000000000
#             if np.abs(self.change_acc) < thr:
#                 print('Center')
#             elif self.change_acc > thr:
#                 print('Left')
#                 for i in range(5):
#                     pyautogui.scroll(-200)
#             elif self.change_acc < thr:
#                 for i in range(5):
#                     pyautogui.scroll(200)
#                 print('Right')
#             self.change_acc = 0
#             self.last_x = 150
#         else:
#             self.change_acc += np.sign(self.last_x - x)*(1.8**(np.abs(self.last_x - x)))
#
#         self.frame_count += 1
#         # r = cv2.erode(r[1] ,np.ones((5,5),dtype=np.uint8),iterations=5)
#         cv2.imshow('f', r)
#
#     def dun(self):
#         self.read_frame()
#         try:
#             eyes_box, _ = self.extract_features()
#         except Exception as e:
#             sys.exit(-1)
#         #b,i = self.get_small_area(eyes_box, self.curr_frame)
#         #ok = self.tracker.init(i, b)
#         ok = self.tracker.init(self.curr_frame, eyes_box)
#         while True:
#             self.read_frame()
#             timer = cv2.getTickCount()
#             self.curr_frame.flags.writeable = False
#             ok, eyes_box = self.tracker.update(self.curr_frame)
#             self.curr_frame.flags.writeable = True
#             self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#             if ok:
#                 #self.get_small_area(eyes_box, self.curr_frame)
#                 ImageUtilities.draw_bound(eyes_box, self.curr_frame)
#                 self.ROI = ImageUtilities.crop_image(eyes_box, self.curr_frame)
#             else:
#                 cv2.putText(self.curr_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                             (0, 0, 255),
#                             2)
#             if (time.time() - self.last_detect_time) > 5.0:
#                 #print('re-extract')
#                 try:
#                     temp_eyes_box, temp_roi = self.extract_features()
#                     eyes_box, self.ROI = temp_eyes_box, temp_roi
#                     self.tracker.init(self.curr_frame, eyes_box)
#                 except Exception as e:
#                     print('can\'t detect...')
#             cv2.putText(self.curr_frame, self.tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                         (50, 170, 50), 2)
#             cv2.putText(self.curr_frame, "FPS : " + str(int(self.fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#                         (50, 170, 50), 2)
#             cv2.imshow("Tracking", self.curr_frame)
#             self.process()
#             if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
#                 break
#         self.video_stream.release()
#         cv2.destroyAllWindows()
#
#     def get_small_area(self, bound, image):
#         up = max(0, bound[1] - 20)
#         down = min(image.shape[0] - 1, bound[0] + bound[2] + 20)
#         left = max(0, bound[0] - 20)
#         right = min(image.shape[1] - 1, bound[1] + bound[3] + 20)
#         i = image[up:down, left:right]
#         self.area_to_search = i.copy()
#         b = (20 ,20 ,right-left ,down-up)
#         cv2.imshow('i', cv2.resize(i,(200,200)))
#         return b,i
#
# #EyesTracker().dun()
import time

import cv2
import sys
import numpy as np
import pyautogui as pyautogui

import ImageUtilities
import TrackerUtilits
from eye_classes import decision


class EyesTracker:
    def __init__(self, tracker_type='MIL'):
        self.tracker_type = tracker_type
        self.face_cascade = cv2.CascadeClassifier('Files\\haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('Files\\haarcascade_eye.xml')
        self.tracker = TrackerUtilits.create_tracker(self.tracker_type)
        self.video_stream = cv2.VideoCapture(0)
        self.last_detect_time = 0
        if not self.video_stream.isOpened():
            print("Could not open video")
            sys.exit()
        ok, frame = self.video_stream.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        self.curr_frame = frame
        self.origin = self.curr_frame.copy()
        self.ROI = None
        self.area_to_search = None
        self.frame_count = 0
        self.last_x = 0
        self.screenshot_counter = 0
        self.change_acc = 1
        self.max_change = 1

        self.max_value = -100000000
        self.min_value = 100000000


        cv2.namedWindow('Tracking')
        cv2.createTrackbar('threshold', 'Tracking', 25, 255, self.do_nothing)

    def do_nothing(self):
        pass

    def detect_faces(self, img):
        frame = None
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        cv2.imshow('gray frame', gray_frame)
        coords = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] * i[2] > biggest[3] * biggest[2]:
                    biggest = i
            biggest = np.array([i], np.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None
        for (x, y, w, h) in biggest:
            frame = img[y:y + h, x:x + w]
        return (frame, biggest[-1])

    def detect_eyes(self, img, point, erode=10):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_frame = cv2.equalizeHist(gray_frame)
        eyes = self.eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
        width = np.size(img, 1)  # get face frame width
        height = np.size(img, 0)  # get face frame height
        left_eye, right_eye = None, None
        for (x, y, w, h) in eyes:
            if y > height / 2:
                pass
            eyecenter = x + w / 2  # get the eye center
            if eyecenter < width * 0.5:
                left_eye = ((point[0] + x, point[1] + y + erode, w, h - erode), img[y + erode:y + h - erode, x:x + w])
                return left_eye
            else:
                right_eye = ((point[0] + x, point[1] + y + erode, w, h - erode), img[y + erode:y + h - erode, x:x + w])
                return right_eye
        return None

    def try_extract_features(self, frame):
        if frame is None:
            return None
        face_detection_results = self.detect_faces(frame)
        if face_detection_results is None:
            return None
        face_frame = face_detection_results[0]
        eyes_detection_result = self.detect_eyes(face_frame, face_detection_results[1])
        if eyes_detection_result is not None:
            return eyes_detection_result
        return None

    def extract_features(self):
        i = 0
        while True:
            eyes_detection_result = self.try_extract_features(frame=self.origin)
            if eyes_detection_result is None and i > 15:
                print('Can\'t detect a face please check about lighting condition and try again....')
                raise Exception
            elif eyes_detection_result is not None and i <= 15:
                self.last_detect_time = time.time()
                return eyes_detection_result
            else:
                self.read_frame()
                i += 1

    def read_frame(self):
        ok, frame = self.video_stream.read()
        if not ok:
            print('Can\'t read from the video stream')
            sys.exit(-1)
        self.curr_frame = cv2.flip(frame, 1)
        self.origin = self.curr_frame.copy()

    def process(self):
        # self.decisor.start(eye = self.ROI ,threshold = cv2.getTrackbarPos('threshold', 'Tracking'))
        roi = cv2.resize(self.ROI, (300, 300))
        #roi_without = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        #roi_without[:, :, 0] = np.uint8(0.5 * roi_without[:, :, 0])
        #roi_without = cv2.cvtColor(roi_without, cv2.COLOR_YUV2BGR)
        #cv2.imshow('ROI without light', roi_without)
        cv2.imshow('ROI', roi)
        r = roi.copy()
        r = cv2.bilateralFilter(r, 11, 50, 25)
        r = ImageUtilities.cvt_to_gray(r, enhancing=False)
        r = cv2.equalizeHist(r)
        cv2.imshow('gray', r)
        threshold = cv2.getTrackbarPos('threshold', 'Tracking')
        r = cv2.threshold(r, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        # r = cv2.adaptiveThreshold(r ,255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,max(2*threshold+1,3),8)
        r = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel=np.ones((5, 5), dtype=np.uint8), iterations=4)
        cv2.imshow('filtered roi', r)
        #r = cv2.erode(r ,np.ones((3,10),dtype=np.uint8),iterations=5)

        r, _, x ,is_screen = ImageUtilities.get_biggest_contour(r)
        if is_screen:
            if self.screenshot_counter > 15:
                print('Screen shot')
                im1 = pyautogui.screenshot()
                im2 = pyautogui.screenshot('my_screenshot.png')
                self.change_acc = 0
                self.last_x = 150
                self.screenshot_counter = 0
            else :
                self.screenshot_counter += 1
        elif self.frame_count % 30 == 0:
            #print('stop')
            #print('acc = ',np.int64(self.change_acc))
            #self.max_change = max(np.int64(self.max_change),np.int64(self.change_acc))
            #print('max = ',self.max_change)
            thr1 = 800
            thr2 = -1000

            #print(self.max_value)
            #print(self.min_value)

            if self.change_acc > thr1:
                print('Left')
                for i in range(5):
                    pyautogui.scroll(-200)
            elif self.change_acc < thr2 :
                for i in range(5):
                    pyautogui.scroll(200)
                print('Right')
            else:
                print('Center')


            self.change_acc = 0
            self.last_x = 150
            self.max_value = -100000000
            self.min_value = 100000000

        else:
            self.change_acc += self.last_x - x
            self.max_value = max(self.max_value , self.change_acc)
            self.min_value = min(self.min_value , self.change_acc)

        self.frame_count += 1
        # r = cv2.erode(r[1] ,np.ones((5,5),dtype=np.uint8),iterations=5)
        cv2.imshow('f', r)

    def dun(self):
        self.read_frame()
        try:
            eyes_box, _ = self.extract_features()
        except Exception as e:
            sys.exit(-1)
        #b,i = self.get_small_area(eyes_box, self.curr_frame)
        #ok = self.tracker.init(i, b)
        ok = self.tracker.init(self.curr_frame, eyes_box)
        while True:
            self.read_frame()
            timer = cv2.getTickCount()
            self.curr_frame.flags.writeable = False
            ok, eyes_box = self.tracker.update(self.curr_frame)
            self.curr_frame.flags.writeable = True
            self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ok:
                #self.get_small_area(eyes_box, self.curr_frame)
                ImageUtilities.draw_bound(eyes_box, self.curr_frame)
                self.ROI = ImageUtilities.crop_image(eyes_box, self.curr_frame)
            else:
                cv2.putText(self.curr_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255),
                            2)
                try:
                    temp_eyes_box, temp_roi = self.extract_features()
                    eyes_box, self.ROI = temp_eyes_box, temp_roi
                    self.tracker.init(self.curr_frame, eyes_box)
                except Exception as e:
                    print('can\'t detect...')
            if (time.time() - self.last_detect_time) > 5.0:
                #print('re-extract')
                try:
                    temp_eyes_box, temp_roi = self.extract_features()
                    eyes_box, self.ROI = temp_eyes_box, temp_roi
                    self.tracker.init(self.curr_frame, eyes_box)
                except Exception as e:
                    print('can\'t detect...')
            cv2.putText(self.curr_frame, self.tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)
            cv2.putText(self.curr_frame, "FPS : " + str(int(self.fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)
            cv2.imshow("Tracking", self.curr_frame)
            self.process()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
                break
        self.video_stream.release()
        cv2.destroyAllWindows()

    def get_small_area(self, bound, image):
        up = max(0, bound[1] - 20)
        down = min(image.shape[0] - 1, bound[0] + bound[2] + 20)
        left = max(0, bound[0] - 20)
        right = min(image.shape[1] - 1, bound[1] + bound[3] + 20)
        i = image[up:down, left:right]
        self.area_to_search = i.copy()
        b = (20 ,20 ,right-left ,down-up)
        cv2.imshow('i', cv2.resize(i,(200,200)))
        return b,i

#EyesTracker().dun()

