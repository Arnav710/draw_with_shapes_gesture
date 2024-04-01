# Import packages
import cv2
import mediapipe as mp
import numpy as np

# Import constants and functions
import constants
from button import draw_button

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

# Flag to indicate if the button was clicked
button_clicked = False

# List to store rectangle coordinates
rectangle_list = []

def add_rectangle():
    """
    Add rectangle coordinates to the list with noise added to the top-right corner.

    Parameters:
        winwidth (int): Width of the window.
    """
    rect_size = (100, 100)
    noise_range = (-10, 10)  # Range for noise addition
    
    # Generate noise for the top-right corner
    noise_x = np.random.randint(noise_range[0], noise_range[1] + 1)
    noise_y = np.random.randint(noise_range[0], noise_range[1] + 1)
    
    # Calculate the coordinates for the top-right corner with noise added
    top_right_x = max(winwidth - rect_size[0] - 20 + noise_x, 0)
    top_right_y = max(20 + noise_y, 0)
    
    top_right = (top_right_x, top_right_y)
    
    bottom_left = (top_right[0] + rect_size[0], top_right[1] + rect_size[1])
    rectangle_list.append((top_right, bottom_left))

def draw_rectangles(img):
    # Draw all rectangles stored in the list onto the frame
    for top_right, bottom_left in rectangle_list:
        cv2.rectangle(img, top_right, bottom_left, (0, 255, 0), -1)

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

def index_finger_inside_rectangle(rec, x, y):
    top_right, bottom_left = rec

    x_in_bounds = top_right[0] <= x and x <= bottom_left[0]
    y_in_bounds = top_right[1] <= y and bottom_left[1]

    return x_in_bounds and y_in_bounds

def move_rectangle(x, y):
    global rectangle_list

    top_right, bottom_left = rectangle_list[-1]

    rectangle_width = bottom_left[0] - top_right[0]
    rectangle_height = bottom_left[1] - top_right[1]

    new_top_right = (int(x - rectangle_width / 2), int(y - rectangle_height / 2))
    new_bottom_left = (new_top_right[0] + rectangle_width, new_top_right[1] + rectangle_height)

    rectangle_list[-1] = (new_top_right, new_bottom_left)

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
                print(index_finger_landmark)
                # Index finger position in pixel coordinates
                index_finger_x = int(index_finger_landmark.x * winwidth)
                index_finger_y = int(index_finger_landmark.y * winheight)

                # If the index finger lies inside the last rectangle placed, we can move it through gestures
                if len(rectangle_list) > 0:
                    rec = rectangle_list[-1]
                    if index_finger_inside_rectangle(rec, index_finger_x, index_finger_y):
                        # Move the most recently added rectangle
                        move_rectangle(index_finger_x, index_finger_y)

        # Resize the frame to the desired window size
        resized_frame = cv2.resize(frame, (winwidth, winheight))

        # draw the button
        resized_frame = draw_button(resized_frame)

        # If button is clicked, add rectangle coordinates to the list
        if button_clicked:
            print("Clicked")
            add_rectangle()
            button_clicked = False  # Reset the button clicked flag


        # Draw all rectangles onto the frame
        draw_rectangles(resized_frame)

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
