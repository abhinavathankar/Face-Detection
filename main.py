import cv2
import numpy as mp
import dlib
from math import hypot




def blinking_ratio(eye_points,facial_landmarks): #Calculates blinking ratio of eye
    leftpoint = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    rightpoint = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    horizontal = cv2.line(frame, leftpoint, rightpoint, (255, 0, 0), 2)

    # print(landmarks.part(36)) #print 36th value

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    verticle = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    verticle_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    # print(verticle_length)#lenth of verticle line
    horizontal_lenth = hypot((leftpoint[0] - rightpoint[0]), (leftpoint[1] - rightpoint[1]))
    # print(horizontal_lenth)#lenth of horizontal line

    ratio = horizontal_lenth / verticle_length
    return ratio

def midpoint(p1,p2): #To get the mid point of two points
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2) #int- pixel cannot be half

cap = cv2.VideoCapture(0)   #Captures Video

detector = dlib.get_frontal_face_detector() #gives four face coordinates
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_COMPLEX

while True: #Loop will run until Escape(27) is pressed
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Gray color will save processing time

    faces = detector(gray)
    for face in faces:  #print(faces) gives four face coordinates
        x,y = face.left(),face.top()
        x1,y1 = face.right(),face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0),2) #Draws the rectangle

        #predictor is the object that is going to find the landmarks
        landmarks = predictor(gray, face)

        left_eye_ratio = blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio = blinking_ratio([42,43,44,45,46,47],landmarks)
        avg_blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        print(avg_blinking_ratio)
        if avg_blinking_ratio > 5:
            cv2.putText(frame,"Blinked",(500,30),font,0.5,(0,255,0))


    cv2.imshow("Frame", frame) #Shows on the screen

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
