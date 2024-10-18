import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
import math
import numpy as np

def is_left_of_line(line_start, line_end, point):
    """
    Checks if a point is to the left of a line formed by line_start and line_end.
    
    Args:
        line_start (tuple): (x1, y1) coordinates of the start of the line.
        line_end (tuple): (x2, y2) coordinates of the end of the line.
        point (tuple): (x, y) coordinates of the point to check.
    
    Returns:
        bool: True if the point is to the left of the line, False otherwise.
    """
    x1, y1 = line_start
    x2, y2 = line_end
    x, y = point

    # Compute the cross product
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # If the cross product is positive, the point is to the left
    return cross_product > 0

def check_valid_labels(data):
    """
    Check that the labels are valid (left eye is to the left of heading vector and right eye is to the right)
    """ 
    line_start = data["heading_points"][0]["x"], data["heading_points"][0]["y"]
    line_end = data["heading_points"][1]["x"], data["heading_points"][1]["y"]
    valid = (is_left_of_line(line_start, line_end, (data["right_points"][0]["x"], data["right_points"][0]["y"]))==False) & \
        (is_left_of_line(line_start, line_end, (data["right_points"][1]["x"], data["right_points"][1]["y"]))==False) & \
        (is_left_of_line(line_start, line_end, (data["left_points"][0]["x"], data["left_points"][0]["y"]))==True) & \
        (is_left_of_line(line_start, line_end, (data["left_points"][1]["x"], data["left_points"][1]["y"]))==True)
    if valid:
        return valid, ""
    else:
        return valid, "all left eye points must be to the left of heading, and all right to the right"


def convert_keypoints(data):
    """
    convert dictionary of keypoints to angle / distance between eyes
    return result as dictionary
    """