import cv2
import numpy as np

# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
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
            left_eye = img[y + 10:y + h - 10, x:x + w]
        else:
            right_eye = img[y + 10:y + h - 10, x:x + w]
    # projective for eyes!!!!!!!!!
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


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    cv2.createTrackbar('filter size', 'image', 1, 255, nothing)
    cv2.createTrackbar('sigma1', 'image', 1, 255, nothing)
    cv2.createTrackbar('sigma2', 'image', 1, 255, nothing)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            if eyes[1] is not None:
                threshold = cv2.getTrackbarPos('threshold', 'image')
                filer_size = max(int(2 * cv2.getTrackbarPos('filter size', 'image') + 1), 1)
                right_eye = cv2.resize(eyes[1], (200, 200))
                gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
                # filterd_gray_right_eye = cv2.medianBlur(gray_right_eye,2*int(filer_size/2)+1)
                sigma1 = cv2.getTrackbarPos('sigma1', 'image')
                sigma2 = cv2.getTrackbarPos('sigma2', 'image')
                filterd_gray_right_eye = cv2.bilateralFilter(gray_right_eye, 2 * int(filer_size / 2) + 1, sigma1,
                                                             sigma2)
                binary_right_eye = cv2.threshold(filterd_gray_right_eye, threshold, 255, cv2.THRESH_BINARY)[1]
                # right_non_zero_pixels = cv2.countNonZero(255-binary_right_eye[:,100:])
                # left_non_zero_pixels = cv2.countNonZero(255-binary_right_eye[:, :100])
                # if right_non_zero_pixels > left_non_zero_pixels:
                #     print('right')
                # else:
                #     print('left')
                # binary_right_eye = cv2.adaptiveThreshold(filterd_gray_right_eye ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY ,filer_size ,8)
                cv2.imshow('gray right eye', gray_right_eye)
                cv2.imshow('filtered gray right eye', filterd_gray_right_eye)
                cv2.imshow('binary right eye', binary_right_eye)
                # filterd_gray_right_eye = cv2.medianBlur(right_eye,2*int(filer_size/2)+1)
                # hsv = cv2.cvtColor(filterd_gray_right_eye,cv2.COLOR_BGR2YUV)
                # hsv[:,:,0] = np.uint8(0.5*hsv[:,:,0])
                # cv2.imshow('HSV' ,hsv)

        # if eyes[0] is not None:
        # left_eye = cv2.resize(eyes[0],(200,200))
        # cv2.imshow('left eye',)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
