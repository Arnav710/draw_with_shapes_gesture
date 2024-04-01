import cv2
import numpy as np
import constants

def add_rectangle(rectangle_list):
    """
    Add rectangle coordinates to the list with noise added to the top-right corner.

    Parameters:
        winwidth (int): Width of the window.
    """
    rect_size = (100, 100)
    noise_range = (-10, 10)  # Range for noise addition
    winwidth = constants.WINDOW_WIDTH
    
    # Generate noise for the top-right corner
    noise_x = np.random.randint(noise_range[0], noise_range[1] + 1)
    noise_y = np.random.randint(noise_range[0], noise_range[1] + 1)
    
    # Calculate the coordinates for the top-right corner with noise added
    top_right_x = max(winwidth - rect_size[0] - 20 + noise_x, 0)
    top_right_y = max(20 + noise_y, 0)
    
    top_right = (top_right_x, top_right_y)
    
    bottom_left = (top_right[0] + rect_size[0], top_right[1] + rect_size[1])
    rectangle_list.append((top_right, bottom_left))

def draw_rectangles(img, rectangle_list):
    # Draw all rectangles stored in the list onto the frame
    for top_right, bottom_left in rectangle_list:
        cv2.rectangle(img, top_right, bottom_left, (0, 255, 0), -1)


def move_rectangle(x, y, rectangle_list):
    top_right, bottom_left = rectangle_list[-1]

    rectangle_width = bottom_left[0] - top_right[0]
    rectangle_height = bottom_left[1] - top_right[1]

    new_top_right = (int(x - rectangle_width / 2), int(y - rectangle_height / 2))
    new_bottom_left = (new_top_right[0] + rectangle_width, new_top_right[1] + rectangle_height)

    rectangle_list[-1] = (new_top_right, new_bottom_left)