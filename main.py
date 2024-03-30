import cv2
import mediapipe as mp

# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Initialize video capture
vidcap = cv2.VideoCapture(0)

# Set the desired window width and height
winwidth = 1280
winheight = 720

def draw_button(img):
    button_x = 20
    button_y = 20
    button_width = 200
    button_height = 50

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

    # Increase spacing between letters
    letter_spacing = 1.2

    # Draw text with increased spacing between letters
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA, False)

    return img


# Initialize hand tracking
with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break
            
        # Draw the button
        frame = draw_button(frame)

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

        # Display the resized frame
        cv2.imshow('Hand Tracking', resized_frame)

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
vidcap.release()
cv2.destroyAllWindows()