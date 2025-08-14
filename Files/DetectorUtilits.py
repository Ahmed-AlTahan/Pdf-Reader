import cv2
import numpy as np

def create_caascade_classifiers():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    return face_cascade, eye_cascade

# if left_eye is not None and right_eye is not None:
            #     #cv2.imshow('img' ,img)
            #     print('fetch two')
            #     right_eye_bound = right_eye[0]
            #     left_eye_bound = left_eye[0]
            #     two_eyes_bound = (left_eye_bound[0]
            #                       ,left_eye_bound[1]
            #                       ,right_eye_bound[0]+right_eye_bound[2]-left_eye_bound[0]
            #                       ,right_eye_bound[1]+right_eye_bound[3]-left_eye_bound[1])
            #     twe_eyes_image = img[two_eyes_bound[1]-point[1]:two_eyes_bound[1]-point[1]+two_eyes_bound[3]
            #     ,two_eyes_bound[0]-point[0]:two_eyes_bound[0]-point[0]+two_eyes_bound[2]]
            #     #cv2.imshow('two eyes' ,img[two_eyes_bound[1]-point[1]:two_eyes_bound[1]-point[1]+two_eyes_bound[3]
            #     #,two_eyes_bound[0]-point[0]:two_eyes_bound[0]-point[0]+two_eyes_bound[2]])
            #     #cv2.waitKey(0)
            #     return two_eyes_bound,twe_eyes_image
        # projective for eyes!!!!!!!!!


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def extract_feature(frame, face_cascade, eye_cascade):
    if frame is None:
        print('None frame')
        return None
    d = detect_faces(frame, face_cascade)
    if d is None:
        print('None face')
        return None
    face_frame = d[0]
    if face_frame is not None:
        eyes = detect_eyes(face_frame, eye_cascade, d[1])
        if eyes is not None:
            return eyes[0], eyes[1]

    return None



