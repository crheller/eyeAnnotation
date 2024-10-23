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
    Rewrite for ellipse data
    """
    if len(data.keys())==0:
        return True, "empty frame"
    line_start = data["heading_points"][0]["x"], data["heading_points"][0]["y"]
    line_end = data["heading_points"][1]["x"], data["heading_points"][1]["y"]
    valid = (is_right_of_line(line_start, line_end, (data["right_eye_x_position"], data["right_eye_y_position"]))==True) & \
        (is_right_of_line(line_start, line_end, (data["left_eye_x_position"], data["left_eye_y_position"]))==False)
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

    # if angleRads < 0:
    #     angleRads += 2*math.pi

    return -1 * angleRads
    

def package_annotations(data, img_path, msg=""):
    """
    New function 22.10.24 to handle new annotations -- which are ellipses for the eyes instead of keypoints
    """
    # convert back to raw image pixels (since we upsample for the Fabric.js canvas)
    img = np.array(Image.open(img_path))
    conversion_factor_x = img.shape[0] / data["canvasWidth"]
    conversion_factor_y = img.shape[1] / data["canvasWidth"]
    
    heading_vec = np.array([data["heading_points"][1]["x"]-data["heading_points"][0]["x"], data["heading_points"][1]["y"]-data["heading_points"][0]["y"]])
    heading_angle = calculateAngle(heading_vec, [1, 0]) # sign flip because y-axis is flipped

    # eye angles should be saved relative to heading
    # to do this, we make a mock vector
    left_x = math.cos(np.deg2rad(data["left_eye_angle"])) 
    left_y = -math.sin(np.deg2rad(data["left_eye_angle"])) 
    right_x = math.cos(np.deg2rad(data["right_eye_angle"])) 
    right_y = -math.sin(np.deg2rad(data["right_eye_angle"]))

    left_vec = np.array([left_x, left_y])
    right_vec = np.array([right_x, right_y])
    left_eye_angle = calculateAngle(left_vec, heading_vec)
    right_eye_angle = calculateAngle(right_vec, heading_vec)

    left_eye_x_position = data["left_eye_x_position"] * conversion_factor_x
    left_eye_y_position = data["left_eye_y_position"] * conversion_factor_y
    right_eye_x_position = data["right_eye_x_position"] * conversion_factor_x
    right_eye_y_position = data["right_eye_y_position"] * conversion_factor_y
    data = {
        "left_eye_angle": left_eye_angle,
        "right_eye_angle": right_eye_angle,
        "heading_angle": heading_angle,
        "left_eye_x_position": left_eye_x_position,
        "left_eye_y_position": left_eye_y_position,
        "right_eye_x_position": right_eye_x_position,
        "right_eye_y_position": right_eye_y_position,
        "yolk_x_position": data["heading_points"][0]["x"] * conversion_factor_x,
        "yolk_y_position": data["heading_points"][0]["y"] * conversion_factor_y,
        "right_eye_missing": 0,
        "left_eye_missing": 0,
        "both_eyes_missing": 0
    }

    return data