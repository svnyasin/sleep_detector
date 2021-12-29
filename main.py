import face_recognition
import cv2
import os
from keras.backend import print_tensor
import numpy as np
from keras.models import load_model
from datetime import datetime   


now = datetime.now()
start_time_str = now.strftime("%H:%M:%S")
start_time = datetime.strptime(start_time_str, "%H:%M:%S")

top_uyuma_sayisi = 0




leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
name = "Unknown"

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


# Load a first sample picture and learn how to recognize it.
yasin_seven_image = face_recognition.load_image_file("yasin_seven.jpg")
yasin_seven_face_encoding = face_recognition.face_encodings(yasin_seven_image)[0]

# Load a first sample picture and learn how to recognize it.
ogulcan_galata_image = face_recognition.load_image_file("ogulcan_galata.jpeg")
ogulcan_galata_face_encoding = face_recognition.face_encodings(ogulcan_galata_image)[0]

# Load a first sample picture and learn how to recognize it.
oguzhan_yilmaz_image = face_recognition.load_image_file("oguzhan_yilmaz.jpeg")
oguzhan_yilmaz_face_encoding = face_recognition.face_encodings(oguzhan_yilmaz_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    yasin_seven_face_encoding,
    ogulcan_galata_face_encoding,
    oguzhan_yilmaz_face_encoding

]
known_face_names = [
    "Yasin Seven",
    "Ogulcan Galata",
    "Oguzhan Yilmaz"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
isFirst=True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    height,width = frame.shape[:2] 


    left_eye = leye.detectMultiScale(frame)
    right_eye =  reye.detectMultiScale(frame)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom ), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )

        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred_x = model.predict(r_eye)
        rpred=np.argmax(rpred_x,axis=1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )

        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred_x = model.predict(l_eye)
        lpred=np.argmax(lpred_x,axis=1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
            break

    if(rpred[0]==0 and lpred[0]==0):
        if (score <= 30):
            score=score+1
        print(name, " ", str(score))
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        if(score<=0):
            score=0  
            print(name, " ", str(score))

        else:
            score=score-2
            print(name, " ", str(score))

            
        
            
    if(score>24):

        if(isFirst):
            top_uyuma_sayisi=top_uyuma_sayisi+1
            print(top_uyuma_sayisi)
            isFirst=False

            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                print("uyuma")
                
                    
            except:  
                pass
        
    else:
        isFirst = True
          

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        now = datetime.now()
        finish_time = now.strftime("%H:%M:%S")
        finish_time = datetime.strptime(finish_time, "%H:%M:%S")
        sure = str(finish_time-start_time)
        f = open("log.txt", "a")
        f.write("\n----------------------------------------------------------------------------------")

        f.write("\nIsim :" + name + "\nSistem baslangic zamani "+ start_time_str+"\nSistem calisma suresi: " + sure + "\nUyuma sayisi :" + str(top_uyuma_sayisi))
        print(sure)
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()