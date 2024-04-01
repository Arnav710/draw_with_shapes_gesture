# Import packages
import cv2
import mediapipe as mp
import numpy as np
import time

# Import constants and functions
import constants
from button import draw_button
from shapes import add_rectangle, draw_rectangles, move_rectangle

# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Initialize video capture
vidcap = cv2.VideoCapture(constants.WEBCAM_SRC)

# Set the desired window width and height
winwidth = constants.WINDOW_WIDTH
winheight = constants.WINDOW_HEIGHT

# Button dimensions
button_x = constants.BUTTON_TOP_RIGHT_X
button_y = constants.BUTTON_TOP_RIGHT_Y
button_width = constants.BUTTON_WIDTH
button_height = constants.BUTTON_HEIGHT

# Flags
button_clicked = False
move_rectangle_enabled = False
start_time = None

# List to store rectangle coordinates
rectangle_list = []

def on_mouse(event, x, y, flags, param):
    global button_clicked
    # when clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        global button_x, button_y, button_width, button_height
        # if the click occurs over the button
        if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
            # and the button has not been clicked already
            if not button_clicked:
                button_clicked = True

def finger_inside_rectangle(rec, x, y):
    top_right, bottom_left = rec

    x_in_bounds = top_right[0] <= x and x <= bottom_left[0]
    y_in_bounds = top_right[1] <= y and bottom_left[1]

    return x_in_bounds and y_in_bounds


# Set up the mouse callback
cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', on_mouse)

# Initialize hand tracking
with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        # flip the frame along the y axis
        frame = cv2.flip(frame,1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = hands.process(rgb_frame)

        # Draw landmarks on the frame
        if processFrames.multi_hand_landmarks:
            index_finger_landmark = None
            for lm in processFrames.multi_hand_landmarks:
                mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                
                index_finger_landmark = lm.landmark[8]
                middle_finger_landmark = lm.landmark[12]

                # Index finger position in pixel coordinates
                index_finger_x = int(index_finger_landmark.x * winwidth)
                index_finger_y = int(index_finger_landmark.y * winheight)

                # Middle finger position in pixel coordinates
                middle_finger_x = int(middle_finger_landmark.x * winwidth)
                middle_finger_y = int(middle_finger_landmark.y * winheight)

                # If the index finger lies inside the last rectangle placed, we can move it through gestures
                if len(rectangle_list) > 0:
                    rec = rectangle_list[-1]

                    if finger_inside_rectangle(rec, index_finger_x, index_finger_y) \
                        and finger_inside_rectangle(rec, middle_finger_x, middle_finger_y):
                        if start_time == None:
                            start_time = time.time()
                        elif time.time() - start_time >= 3:                  
                            move_rectangle_enabled = not move_rectangle_enabled
                            start_time = time.time()
                            print("move rectangle", move_rectangle_enabled)
                        
                    else:
                        start_time = None
                    

                    if move_rectangle_enabled:
                        # Move the most recently added rectangle
                        move_rectangle(index_finger_x, index_finger_y, rectangle_list)

        # Resize the frame to the desired window size
        resized_frame = cv2.resize(frame, (winwidth, winheight))

        # draw the button
        resized_frame = draw_button(resized_frame)

        # If button is clicked, add rectangle coordinates to the list
        if button_clicked:
            print("Clicked")
            add_rectangle(rectangle_list)
            button_clicked = False  # Reset the button clicked flag


        # Draw all rectangles onto the frame
        draw_rectangles(resized_frame, rectangle_list)

        # Display the resized frame
        cv2.imshow('Hand Tracking', resized_frame)

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if the button is clicked (mouse event)
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break

# Release the video capture and close windows
vidcap.release()
cv2.destroyAllWindows()