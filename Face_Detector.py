import cv2

#Loading our pre-trained data on face frontals from opencv 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Import an image into opencv
# img = cv2.imread('rdj.jpg')
# img = cv2.imread('banner-image.png')
img = cv2.imread('Robert-Downey-Jr-Matt-Damon.jpg')

#Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) #this returns a list of coordinates of the faces in the image
print(face_coordinates) #this prints the coordinates of the faces in the image x,y,w,h
#(x,y) is the top left corner of the face and w,h is the width and height of the face

#Draw rectangles around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)



#Show the image
cv2.imshow('Face detector', img)

#Wait for a key to be pressed before closing the image
cv2.waitKey()
print("Code completed")

