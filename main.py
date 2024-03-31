import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Initialize video capture
vidcap = cv2.VideoCapture(0)

# Set the desired window width and height
winwidth = 1280
winheight = 720

# Button dimensions
button_x = 20
button_y = 20
button_width = 200
button_height = 50

# Flag to indicate if the button was clicked
button_clicked = False

# List to store rectangle coordinates
rectangle_list = []

def draw_button(img):
    # Draw white filled rectangle with black border
    cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), (255, 255, 255), -1)
    cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), (0, 0, 0), 2)

    # Add text inside the button
    text = "Add Rectangle"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Adjust font size and thickness
    font_scale = 0.8
    thickness = 2

    # Get text size and position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = button_x + (button_width - text_size[0]) // 2
    text_y = button_y + (button_height + text_size[1]) // 2

    # Draw text with increased spacing between letters
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA, False)

    return img

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

# Set up the mouse callback
cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', on_mouse)

# Initialize hand tracking
with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = hands.process(rgb_frame)

        # Draw landmarks on the frame
        if processFrames.multi_hand_landmarks:
            for lm in processFrames.multi_hand_landmarks:
                mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)

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
