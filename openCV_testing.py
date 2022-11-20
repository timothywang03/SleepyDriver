import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def cut_eyebrows(img):		#essentially cuts out eyebrow from frame (incase program detects it as an eye)
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def detect_eyes(img, cascade):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		#grayscale img
	eyes = cascade.detectMultiScale(gray_img, 1.3, 5)		#detect eyes
	width = np.size(img, 1)		#get width and height of image
	height = np.size(img, 0)
	left_eye = None
	right_eye = None
	for (x,y,w,h) in eyes:
		if y>height/2:		#make sure eye is in top half of frame
			pass
		eyecenter = x+w/2		#get center of eye
		if eyecenter<width/2:
			left_eye = img[y:y+h,x:x+w]		#classify as left or right eye
		else:
			right_eye = img[y:y+h,x:x+w]
	return left_eye, right_eye


def detect_faces(img, cascade):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	coords = cascade.detectMultiScale(gray_img, 1.3, 5)		#detect face
	if len(coords) > 1:					#detect the largest face ONLY and return that frame
		biggest = (0,0,0,0)
		for i in coords:
			if i[3] > biggest[3]:
				biggest=i
		biggest = np.array([i],np.int32)
	elif len(coords)==1:
		biggest=coords
	else:
		return None
	for (x,y,w,h) in biggest:
		frame = img[y:y+h, x:x+w]
	return frame


def blob_process(img, threshold, detector):			#isolate the pupil in the eye
	gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)		#convert to black and white w threshold
	img = cv2.erode(img,None, iterations=2)			#get rid of noise in image
	img = cv2.dilate(img, None, iterations=4)
	img = cv2.medianBlur(img, 5)
	keypoints = detector.detect(img)
	return keypoints


def nothing(x):
	pass


def main():
	cap = cv2.VideoCapture(0)		#turn on video
	cv2.namedWindow('image')
	cv2.createTrackbar('threshold', 'image', 0, 255, nothing)		#creates top trackbar
	while True:
		_, frame = cap.read()	 #gets image/frame from feed
		face_frame = detect_faces(frame, face_cascade)		#detect face
		if face_frame is not None:
			eyes = detect_eyes(face_frame, eye_cascade)		#detect eyes
			for eye in eyes:
				if eye is not None:
					threshold = r = cv2.getTrackbarPos('threshold', 'image')	#get threshold value
					print('threshold: '+ str(threshold))
					eye = cut_eyebrows(eye)
					keypoints = blob_process(eye, threshold, detector) 		#isolate pupil
					eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		cv2.imshow('image', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
