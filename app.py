import sys
sys.path.append("/home/charlie/code/eyeAnnotation")
from flask import Flask, render_template, request, jsonify, url_for, session
import json
import os
from loader.data_loader import load_new_image
from annotations.helpers import package_annotations, check_valid_labels, plot_training_distribution
import subprocess
from settings import IMG_DIR, ANNOT_DIR
import datetime
import glob

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # You should replace this with a secure key in production

# Serve the image when the button is clicked
def load_image(uid):
    path = load_new_image()
    static_path = f"image{uid}.png"
    subprocess.run(['cp', path, f"static/{static_path}"], check=True)
    return path, static_path

@app.route('/')
def index():
    now = datetime.datetime.now()
    uid = now.strftime('%y%m%d_%H%M%S')
    img_tmp, img_path = load_image(uid)
    distro_json = plot_training_distribution()
    print(session)
    session["IMG_TMP_PATH"] = img_tmp
    session["IMG_PATH"] = img_path
    print(session)
    return render_template('index.html', image_path=url_for('static', filename=session["IMG_PATH"]), distro_json=distro_json, demo_path=url_for('static', filename="eye_example.png"))


@app.route('/save_annotation', methods=['POST'])
def save_annotation():

    data = request.json

    valid, msg = check_valid_labels(data)
    if valid==False:
         samepath = url_for('static', filename=session["IMG_PATH"])
         return jsonify({"message": f"Error: {msg}", "image_path": f"{samepath}"})
    elif (valid) & (msg=="empty frame"):
        # bad frame, load new data without saving anything
        ftodelete = glob.glob(f'static/{session["IMG_PATH"]}')
        subprocess.Popen(['rm'] + ftodelete)
        subprocess.Popen(['rm'] + [session["IMG_TMP_PATH"]])
    
        now = datetime.datetime.now()
        uid = now.strftime('%y%m%d_%H%M%S')
        img_tmp, img_path = load_image(uid) # returns the full annotation path
        session["IMG_TMP_PATH"] = img_tmp
        session["IMG_PATH"] = img_path
        static_path = url_for('static', filename=session["IMG_PATH"])
        return jsonify({"image_path": f"{static_path}"})
    
    # convert the key point annotations to eye distance / angle (the outputs of the model)
    processed_data = package_annotations(data, session["IMG_TMP_PATH"], msg)

    # Save annotation to a file
    name = os.path.basename(session["IMG_TMP_PATH"]).replace(".png", ".json")
    savefile = os.path.join(ANNOT_DIR, name)
    if not os.path.exists(savefile):
        with open(savefile, 'w') as f:
            json.dump([], f)  # Create empty list if file doesn't exist
   
    with open(savefile, 'r') as f:  
        annotations = json.load(f)
    
    annotations.append(processed_data)

    with open(savefile, 'w') as f:
        json.dump(annotations, f)

    # move image to proper folder
    img_save_path = os.path.join(IMG_DIR, os.path.basename(session["IMG_TMP_PATH"]))
    subprocess.Popen(['mv', session["IMG_TMP_PATH"], img_save_path])

    # reload page (which loads a new image)
    # subprocess.Popen(['touch', 'app.py'])

    # get a new image / remove old one
    ftodelete = glob.glob(f'static/{session["IMG_PATH"]}')
    subprocess.Popen(['rm'] + ftodelete)
    now = datetime.datetime.now()
    uid = now.strftime('%y%m%d_%H%M%S')
    img_tmp, img_path = load_image(uid) # returns the full annotation path
    session["IMG_TMP_PATH"] = img_tmp
    session["IMG_PATH"] = img_path

    # somehow pass this with jsonify back to the html so that the html callback can dynamically reserve the (new) image
    # but also need to keep track of this filename so that we know how to save the annotations...
    distro_json = plot_training_distribution()
    path = url_for('static', filename=f"{img_path}")
    
    return jsonify({"image_path": f"{path}", "distro_json": distro_json})

@app.route('/window_closed', methods=['POST'])
def window_closed():
    # delete the tmp frame and the static frame if it was not annotated
    ftodelete = glob.glob(f'static/{session["IMG_PATH"]}')
    subprocess.Popen(['rm'] + ftodelete)
    subprocess.Popen(['rm'] + [session["IMG_TMP_PATH"]])
    print("Browser window was closed!")
    return '', 204  # Return a no-content response


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)