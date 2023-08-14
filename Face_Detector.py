import cv2

#Loading our pre-trained data on face frontals from opencv 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Note: the Haar Cascade algorithm is more based on speed than accuracy
#So it has 90% accuracy and is very fast

#Import an image into opencv
# img = cv2.imread(r"images\banner-image.png")
# img = cv2.imread(r"images\rdj.jpg")
# img = cv2.imread(r"images\Screenshot 2023-06-15 151743.png")
# img = cv2.imread(r"images\groupimage.jpg")

#To capture video from webcam 
webcam = cv2.VideoCapture(0) #0 is the default webcam
#webcam = cv2.VideoCapture("video.mp4") #this is to capture video from a video file

#Video is a series of images, so we need to iterate over the frames (images)
#Iterate forever over frames
while True:
    #this reads the current frame 
    successful_frame_read, frame = webcam.read() #this returns a tuple of a boolean (if frame was read) and the frame


# #Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) #this returns a list of coordinates of the faces in the image
    # print(face_coordinates) #this prints the coordinates of the faces in the image x,y,w,h
    #(x,y) is the top left corner of the face and w,h is the width and height of the face

    #Draw rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        # x1= int(x+w/2)
        # y1= int(y+h/2)
        # cv2.circle(img, (x1,y1), int(h/2), (0,255,0), 2) #this takes the image, the top left corner, the bottom right corner, the color of the rectangle and the thickness of the rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) #this takes the image, the top left corner, the bottom right corner, the color of the rectangle and the thickness of the rectangle
        #also note that the color is in BGR format instead of RGB (in opencv)


    # #Show the image
    cv2.imshow('Face detector', frame)

    #Wait for a key to be pressed before closing the image
    #waitkey has default as 0 which means it waits infinitively for a key to be pressed before closing the image
    #however with a video, this means showing only one frame
    #so we add a parameter of 1 to wait for 1 millisecond before switching to the next frame
    key= cv2.waitKey(1) #adding a parameter of 1 makes the image wait for only 1 millisecond before switching to the next frame
    if key==65 or key==97: #this is to exit the program when the letter a or A is pressed
        break

#Release the VideoCapture object to free up the memory
webcam.release()
print("Code completed")

