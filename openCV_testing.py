import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def cut_eyebrows(img):
    """Given an image, cut out the eyebrows from the eyes"""
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def detect_eyes(img, cascade):
    """Given an image and a cascade effect, return a nested array of coordinates for respective eyes"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_img, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2
        if eyecenter < width / 2:
            left_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
        else:
            right_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)

    return left_eye, right_eye


def detect_faces(img, cascade):
    "Given an image and a face cascading effect, return the coordinates of the face"
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_img, 1.3, 5)
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


def blob_process(img, threshold, detector):
    """Using blur processing, get the pupils of the eyes --threshold is togglable"""
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    # marks the last time that both eyes were detected
    time_since_attention = None

    while True:

        # attention refers to whether or not both eyes are detected
        # the driver is not paying attention when attention is false
        attention = True
        current_time = time.time()

        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)

            for eye in eyes:
                if eye is not None:
                    # threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    threshold = r = 38    # timothy threshold value

                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 255, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if eyes[0] is None and eyes[1] is None:
                if attention is True:
                    time_since_attention = time.time()
                attention = False
            else:
                attention = True

        # outputs a boolean based on whether or not the car should be moving
        if time_since_attention is not None and current_time - time_since_attention > 2:
            print(False)
        else:
            print(True)

        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread('test_image.jpeg')
    main()
