import cv2

# Setting up the video camera with source 0 - inbuilt webcam
cam = cv2.VideoCapture(0)

# Setting the width and height
cam.set(3, 1280)
cam.set(4, 720)

while True:
    success, img = cam.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1 ms delay