# We will:
# 1. Load the image.
# 2. Convert it to grayscale.
# 3. Detect the face.
# 4. Detect the eyes within the face region.
# 5. Crop and resize the eye regions.
# 6. Normalize the pixel values .

import cv2
import os

save_dir="D:\\Stuff\\Data Science\\DDD\\EyeDataset\\drowsy"
eye_count=1

path="DriverDrowsinessDataset(DDD)\\Drowsy"
files=os.listdir(path)
width, height=800, 600

# cv2.CascadeClassifier(): This function loads a Haar Cascade classifier
# which is an XML file containing a trained model for detecting specific features like faces or eyes in an image.

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

for file in files:
    img_path=os.path.join(path, file)
    img=cv2.imread(img_path)
    img=cv2.resize(img,(width, height))
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # After converting them to grayscaled images, detect faces and then detect the eyes from faces using haarcascade
    faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
    # Higher minNeighbors = Fewer false positives but might miss some faces/eyes, a lower value will less accurate and detect more
    for (x,y,w,h) in faces[:15]:
        face_region=gray_img[y:y+h, x:x+w]
        # To show the faces in a rectangular box
        # cv2.rectangle(gray_img, (x,y), (x+w,y+h), (255,0,0),2)
        # cv2.imshow("Detected Face", gray_img)
        # cv2.waitKey()
        eyes=eye_cascade.detectMultiScale(face_region)
        eyes=sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle for eyes on original image (not on face_region)
            # Add face coordinates (x,y) as offset since eye coordinates are relative to face
            if ey>face_region.shape[0]//2:
                continue
            #Limit the number of eyes detected to 2
            cv2.rectangle(gray_img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            eye = face_region[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (64, 64))
            eye_normal = cv2.normalize(eye.astype('float32'), None, 0, 255, cv2.NORM_MINMAX)
            eye_normal=eye_normal.astype('uint8')
            # the normalize function converts the resized eye image to float32 for better precision, ensures accurate scaling
            # the normalized values are from 0 to 255
            # uses min_max normalization
            # cv2.imshow('Detected eyes', gray_img)
            # cv2.imshow("Eye Only", eye_normal)
            filename=f'Drowsy{eye_count}.jpg'
            save_path=os.path.join(save_dir,filename)
            cv2.imwrite(save_path, eye_normal)
            eye_count+=1
            cv2.waitKey()
cv2.destroyAllWindows() 