import cv2
import numpy as np
import dlib
from curve_fitting import curve
from imutils import face_utils

path='./results'
img_path='image.jpg'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        l=[]

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i in range(5, 12):                          #loop for storing coordinates of jaw
            curr_cordi=(shape[i][0], shape[i][1])
            l.append(curr_cordi)

        cur=np.array(curve(np.array(l)), np.int32)      # calling function to find proper fitting curve


        for i in range(len(cur)-1):                          #loop for drawing jaw line
            curr_cordi=(cur[i][0], cur[i][1])
            next_cordi=(cur[i+1][0], cur[i+1][1])
            cv2.line(image, curr_cordi, next_cordi, (0, 0, 255), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image

        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


cv2.imwrite(path+"/"+img_path, image)
