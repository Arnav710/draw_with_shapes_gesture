import cv2

import constants

def draw_button(img):

    button_x = constants.BUTTON_TOP_RIGHT_X
    button_y = constants.BUTTON_TOP_RIGHT_Y
    button_width = constants.BUTTON_WIDTH
    button_height = constants.BUTTON_HEIGHT

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