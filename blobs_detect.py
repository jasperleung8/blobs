import cv2
import numpy as np
img = cv2.imread("blobs.jpg")

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 50

params.filterByCircularity = False

params.filterByConvexity = True
params.minConvexity = 0.2

params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(img,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number = len(keypoints)
text = "Number of small non-circular Blobs :" + str(len(keypoints))
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,255),2)

cv2.imshow("Detect blobs",blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
