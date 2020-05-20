import cv2
import numpy as mp
import dlib

def midpoint(p1,p2): #To get the mid point of two points
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2) #int- pixel cannot be half

cap = cv2.VideoCapture(0)   #Captures Video

detector = dlib.get_frontal_face_detector() #gives four face coordinates
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True: #Loop will run until Escape(27) is pressed
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Gray color will save processing time

    faces = detector(gray)
    for face in faces:  #print(faces) gives four face coordinates
        #x,y = face.left(),face.top()
        #x1,y1 = face.right(),face.bottom()
        #cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0),2) #Draws the rectangle
        #predictor is the object that is going to find the landmarks

        landmarks = predictor(gray, face)
        leftpoint = (landmarks.part(36).x, landmarks.part(36).y)
        rightpoint = (landmarks.part(39).x, landmarks.part(39).y)
        horizontal = cv2.line(frame, leftpoint, rightpoint, (255, 0, 0), 2)

        #print(landmarks.part(36)) #print 36th value

        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        verticle = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    cv2.imshow("Frame", frame) #Shows on the screen

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
