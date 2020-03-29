import cv2

facexml = cv2.CascadeClassifier('face.xml')
webcam = cv2.VideoCapture(0)        # open web cam
sabzeh = cv2.imread("Sabzeh.png", -1)

while True:
	ret, frame = webcam.read()      # get frame
  
	if frame is None:
		break
		
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # rgb to gray
	gray = cv2.equalizeHist(gray)
		
	faces = facexml.detectMultiScale(frame)               # use xml

	# if len(faces) > 1: 
	# 	raise ValueError('only one face !!!   But: ' + str(len(faces)))

	for (x,y,w,h) in faces:
		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)  # draw a rectangle

		font = cv2.FONT_HERSHEY_PLAIN
		coverResized = cv2.resize(sabzeh, (w+10, h))
		# cv2.imshow('cover', coverResized)

		y_offset=y-130
		y1, y2 = y_offset, y_offset + coverResized.shape[0]
		x1, x2 =  x, x + coverResized.shape[1]

		alpha_s = coverResized[:, :, 3] / 255.0
		alpha_l = 1.0 - alpha_s
		
		for c in range(0, 3):
			frame[y1:y2, x1:x2, c] = (alpha_s * coverResized[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

		cv2.imshow("kaleh sabzehi 1399/1/2    :) ", frame)        # show 

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):              # exit by press key Q
		break
 

cv2.destroyAllWindows()              # close wiindows
