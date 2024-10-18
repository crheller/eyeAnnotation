"""
One off script to fix annotations, if needed
"""
import json
from settings import IMG_DIR
import os

count = 0
for f in os.listdir(IMG_DIR):
    if (f!="images") & (f!="tmp"):
        with open(os.path.join(IMG_DIR, f)) as fh:
            data = json.load(fh)[0]

        # add new fields
        data["left_eye_missing"] = 0
        data["right_eye_missing"] = 0
        data["both_eyes_missing"] = 0
        
        with open(os.path.join(IMG_DIR, f), "w") as fh:
            json.dump([data], fh)

        count+=1
        
print(f"count: {count}")