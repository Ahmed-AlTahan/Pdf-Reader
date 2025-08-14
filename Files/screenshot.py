import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

face_cascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"haarcascade_eye.xml")
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints


def nothing(x):
    pass

def count_white_pixels(img) :
    n_white_pix = np.sum(img == 255)
    return n_white_pix



def main():
    cap = cv2.VideoCapture(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    #print(cap.get(cv2.cv.CV_CAP_PROP_FPS))


    cv2.namedWindow('image')
    # Bar
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    frame_counter = 0
    res_right = 0
    res_left = 0
    res_center = 0
    close = 0
    while True:

        if frame_counter > 10 :
            frame_counter = 0
            res_right = 0
            res_left = 0
            res_center = 0
            close = 0

        frame_counter = frame_counter + 1

        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        face_frame = detect_faces(frame, face_cascade)

        result = ""
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            close_eye = 0
            for i in range(len(eyes)) :

                if eyes[i] is None:
                    close_eye = close_eye + 1

                if eyes[i] is not None:
                    eye = eyes[i]
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')

                    eye = cut_eyebrows(eye)

                    ##################


                    if i == 0 :
                        right = eye
                        right = cv2.resize(right, (300, 300))
                        cv2.imshow("right" , right)
                        right_gray = cv2.cvtColor(right , cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(right_gray, threshold, 255, cv2.THRESH_BINARY_INV)
                        #cv2.imshow("thre", thresh)
                        kernel = np.ones((5, 5), np.uint8)
                        erosion = cv2.erode(thresh, kernel , iterations = 4)
                        cv2.imshow("binary_erode" , erosion)


                        left_image = erosion[: , 0:120]
                        right_image = erosion[: , 220:]
                        center_image = erosion[: , 120:200]

                        #cv2.imshow("left_image" , left_image)
                        #cv2.imshow("right_image" , right_image)
                        #cv2.imshow("center_image" , center_image)


                        cnt_left_image = count_white_pixels(left_image)
                        cnt_right_image = count_white_pixels(right_image)
                        cnt_center_image = count_white_pixels(center_image)

                        maximum = max(cnt_center_image , cnt_right_image , cnt_left_image)
                        if maximum == cnt_center_image :
                            print("center")
                            result = "center"
                        if maximum == cnt_right_image :
                            print("right")
                            result = "right"
                        if maximum == cnt_left_image :
                            print("left")
                            result = "left"






                    # else :
                    #     left = eye
                    #     left = cv2.resize(left, (300, 300))
                    #     cv2.imshow("left", left)
                    #     ret, thresh = cv2.threshold(left, 127, 255, cv2.THRESH_BINARY)

                    #################


                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if close_eye == 2 :
                close = 1

        if result == "right" :
            res_right = res_right + 1
        if result == "left":
            res_left = res_left + 1
        if result == "center":
            res_center = res_center + 1

        if frame_counter == 10 :
            if res_right :
                result = "right"
            elif res_left :
                result = "left"
            else :
                result = "center"
            frame = cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if close == 1 :
                frame = cv2.putText(frame, "Screenshot", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()