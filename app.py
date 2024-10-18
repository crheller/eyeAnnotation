import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
from flask import Flask, render_template, request, jsonify, url_for, make_response, send_from_directory
# from helpers import get_pos_angle
import json
import os
from loader.data_loader import load_new_image
from annotations.helpers import convert_keypoints, check_valid_labels
import subprocess
from settings import IMG_DIR
import datetime
import glob

app = Flask(__name__)

# Global variables
global IMAGE_PATH
IMAGE_PATH = 'image.png'

# the path to the actual data image / where we will save annotations
global IMAGE_TMP_PATH 
IMAGE_TMP_PATH = '' # gets updated on page load


# Serve the image when the button is clicked
def load_image(uid):
    path = load_new_image()
    static_path = f"image{uid}.png"
    subprocess.run(['cp', path, f"static/{static_path}"], check=True)
    return path, static_path

@app.route('/')
def index():
    global IMAGE_TMP_PATH 
    IMAGE_TMP_PATH, IMAGE_PATH = load_image("")
    return render_template('index.html', image_path=url_for('static', filename=IMAGE_PATH), demo_path=url_for('static', filename="eye_example.png"))


@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    global IMAGE_TMP_PATH, IMAGE_PATH

    data = request.json

    valid, msg = check_valid_labels(data)
    if valid==False:
         samepath = url_for('static', filename=IMAGE_PATH)
         return jsonify({"message": f"Error: {msg}", "image_path": f"{samepath}"})
    
    # convert the key point annotations to eye distance / angle (the outputs of the model)
    processed_data = convert_keypoints(data, msg)

    # Save annotation to a file
    name = os.path.basename(IMAGE_TMP_PATH).replace(".png", ".json")
    savefile = os.path.join(IMG_DIR, name)
    if not os.path.exists(savefile):
        with open(savefile, 'w') as f:
            json.dump([], f)  # Create empty list if file doesn't exist
   
    with open(savefile, 'r') as f:  
        annotations = json.load(f)
    
    annotations.append(processed_data)

    with open(savefile, 'w') as f:
        json.dump(annotations, f)

    # move image to proper folder
    img_save_path = IMAGE_TMP_PATH.replace("/tmp/", "/images/")
    subprocess.Popen(['mv', IMAGE_TMP_PATH, img_save_path])

    # reload page (which loads a new image)
    # subprocess.Popen(['touch', 'app.py'])

    # get a new image / remove old one
    ftodelete = glob.glob(f'static/{IMAGE_PATH}')
    subprocess.Popen(['rm'] + ftodelete)
    now = datetime.datetime.now()
    uid = now.strftime('%y%m%d_%H%M%S')
    IMAGE_TMP_PATH, IMAGE_PATH = load_image(uid) # returns the full annotation path
    # somehow pass this with jsonify back to the html so that the html callback can dynamically reserve the (new) image
    # but also need to keep track of this filename so that we know how to save the annotations...
    path = url_for('static', filename=f"image{uid}.png")
    
    return jsonify({"image_path": f"{path}"})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)