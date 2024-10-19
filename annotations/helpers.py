import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
import math
import numpy as np
from PIL import Image

def is_right_of_line(line_start, line_end, point):
    """
    Checks if a point is to the right of a line formed by line_start and line_end
    Note, this is because Fabric.js inverts the y-axis.
    
    Args:
        line_start (tuple): (x1, y1) coordinates of the start of the line.
        line_end (tuple): (x2, y2) coordinates of the end of the line.
        point (tuple): (x, y) coordinates of the point to check.
    
    Returns:
        bool: True if the point is to the right of the line, False otherwise.
    """
    x1, y1 = line_start
    x2, y2 = line_end
    x, y = point

    # Compute the cross product
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    # If the cross product is positive, the point is to the right
    return cross_product > 0

def check_valid_labels(data):
    """
    Check that the labels are valid (left eye is to the left of heading vector and right eye is to the right)
    """ 
    if (data["left_points"]==[]) & (data["right_points"]!=[]):
        line_start = data["heading_points"][0]["x"], data["heading_points"][0]["y"]
        line_end = data["heading_points"][1]["x"], data["heading_points"][1]["y"]
        valid = (is_right_of_line(line_start, line_end, (data["right_points"][0]["x"], data["right_points"][0]["y"]))==True) & \
            (is_right_of_line(line_start, line_end, (data["right_points"][1]["x"], data["right_points"][1]["y"]))==True)
        if valid:
            return valid, "missing_left"
        else:
            return valid, "all left eye points must be to the left of heading, and all right to the right"
        
    elif (data["left_points"]!=[]) & (data["right_points"]==[]):
        line_start = data["heading_points"][0]["x"], data["heading_points"][0]["y"]
        line_end = data["heading_points"][1]["x"], data["heading_points"][1]["y"]
        valid = (is_right_of_line(line_start, line_end, (data["left_points"][0]["x"], data["left_points"][0]["y"]))==False) & \
            (is_right_of_line(line_start, line_end, (data["left_points"][1]["x"], data["left_points"][1]["y"]))==False)
        if valid:
            return valid, "missing_right"
        else:
            return valid, "all left eye points must be to the left of heading, and all right to the right"
        
    elif (data["left_points"]==[]) & (data["right_points"]==[]):
        return True, "missing_both"

    else:
        line_start = data["heading_points"][0]["x"], data["heading_points"][0]["y"]
        line_end = data["heading_points"][1]["x"], data["heading_points"][1]["y"]
        valid = (is_right_of_line(line_start, line_end, (data["right_points"][0]["x"], data["right_points"][0]["y"]))==True) & \
            (is_right_of_line(line_start, line_end, (data["right_points"][1]["x"], data["right_points"][1]["y"]))==True) & \
            (is_right_of_line(line_start, line_end, (data["left_points"][0]["x"], data["left_points"][0]["y"]))==False) & \
            (is_right_of_line(line_start, line_end, (data["left_points"][1]["x"], data["left_points"][1]["y"]))==False)
        if valid:
            return valid, ""
        else:
            return valid, "all left eye points must be to the left of heading, and all right to the right"


def calculateAngle(l1, l2):

    l1 = l1 / np.linalg.norm(l1)
    l2 = l2 / np.linalg.norm(l2)
    # get sign
    cross_prod = l1[0] * l2[1] - l1[1] * l2[0]

    angleRads = np.sign(cross_prod) * math.acos(l1.dot(l2))

    return -1 * angleRads


def convert_keypoints(data, img_path, msg=""):
    """
    convert dictionary of keypoints to angle / distance between eyes
    return result as dictionary
    """
    # convert back to raw image pixels (since we upsample for the Fabric.js canvas)
    img = np.array(Image.open(img_path))
    conversion_factor_x = img.shape[0] / data["canvasWidth"]
    conversion_factor_y = img.shape[1] / data["canvasWidth"]
    if msg=="":
        left_vec = np.array([data["left_points"][1]["x"]-data["left_points"][0]["x"], data["left_points"][1]["y"]-data["left_points"][0]["y"]])
        right_vec = np.array([data["right_points"][1]["x"]-data["right_points"][0]["x"], data["right_points"][1]["y"]-data["right_points"][0]["y"]])
        heading_vec = np.array([data["heading_points"][1]["x"]-data["heading_points"][0]["x"], data["heading_points"][1]["y"]-data["heading_points"][0]["y"]])
        
        left_eye_angle = calculateAngle(left_vec, heading_vec)
        right_eye_angle = calculateAngle(right_vec, heading_vec)
        heading_angle = calculateAngle(heading_vec, [1, 0])

        midpoint_left = np.array([(data["left_points"][1]["x"]+data["left_points"][0]["x"])/2, (data["left_points"][1]["y"]+data["left_points"][0]["y"])/2])
        midpoint_right = np.array([(data["right_points"][1]["x"]+data["right_points"][0]["x"])/2, (data["right_points"][1]["y"]+data["right_points"][0]["y"])/2])

        data = {
            "left_eye_angle": left_eye_angle,
            "right_eye_angle": right_eye_angle,
            "heading_angle": heading_angle,
            "left_eye_x_position": midpoint_left[0] * conversion_factor_x,
            "left_eye_y_position": midpoint_left[1] * conversion_factor_y,
            "right_eye_x_position": midpoint_right[0] * conversion_factor_x,
            "right_eye_y_position": midpoint_right[1] * conversion_factor_y,
            "yolk_x_position": data["heading_points"][0]["x"] * conversion_factor_x,
            "yolk_y_position": data["heading_points"][0]["y"] * conversion_factor_y,
            "right_eye_missing": 0,
            "left_eye_missing": 0,
            "both_eyes_missing": 0
        }
        return data

    elif msg=="missing_left":
        right_vec = np.array([data["right_points"][1]["x"]-data["right_points"][0]["x"], data["right_points"][1]["y"]-data["right_points"][0]["y"]])
        heading_vec = np.array([data["heading_points"][1]["x"]-data["heading_points"][0]["x"], data["heading_points"][1]["y"]-data["heading_points"][0]["y"]])
        right_eye_angle = calculateAngle(right_vec, heading_vec)
        heading_angle = calculateAngle(heading_vec, [1, 0])
        midpoint_right = np.array([(data["right_points"][1]["x"]+data["right_points"][0]["x"])/2, (data["right_points"][1]["y"]+data["right_points"][0]["y"])/2])
        data = {
            "left_eye_angle": 0,
            "right_eye_angle": right_eye_angle,
            "heading_angle": heading_angle,
            "left_eye_x_position": 0,
            "left_eye_y_position": 0,
            "right_eye_x_position": midpoint_right[0] * conversion_factor_x,
            "right_eye_y_position": midpoint_right[1] * conversion_factor_y,
            "yolk_x_position": data["heading_points"][0]["x"] * conversion_factor_x,
            "yolk_y_position": data["heading_points"][0]["y"] * conversion_factor_y,
            "right_eye_missing": 0,
            "left_eye_missing": 1,
            "both_eyes_missing": 0
        }
        return data
    
    elif msg=="missing_right":
        left_vec = np.array([data["left_points"][1]["x"]-data["left_points"][0]["x"], data["left_points"][1]["y"]-data["left_points"][0]["y"]])
        heading_vec = np.array([data["heading_points"][1]["x"]-data["heading_points"][0]["x"], data["heading_points"][1]["y"]-data["heading_points"][0]["y"]])
        left_eye_angle = calculateAngle(left_vec, heading_vec)
        heading_angle = calculateAngle(heading_vec, [1, 0])
        data = {
            "left_eye_angle": left_eye_angle,
            "right_eye_angle": 0,
            "heading_angle": heading_angle,
            "left_eye_x_position": midpoint_left[0] * conversion_factor_x,
            "left_eye_y_position": midpoint_left[1] * conversion_factor_y,
            "right_eye_x_position": 0,
            "right_eye_y_position": 0,
            "yolk_x_position": data["heading_points"][0]["x"] * conversion_factor_x,
            "yolk_y_position": data["heading_points"][0]["y"] * conversion_factor_y,
            "right_eye_missing": 1,
            "left_eye_missing": 0,
            "both_eyes_missing": 0
        }
        return data

    elif msg=="missing_both":
        data = {
            "left_eye_angle": 0,
            "right_eye_angle": 0,
            "heading_angle": 0,
            "left_eye_x_position": 0,
            "left_eye_y_position": 0,
            "right_eye_x_position": 0,
            "right_eye_y_position": 0,
            "yolk_x_position": 0,
            "yolk_y_position": 0,
            "right_eye_missing": 1,
            "left_eye_missing": 1,
            "both_eyes_missing": 1
        }
        return data