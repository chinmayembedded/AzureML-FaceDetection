import os
import requests
import json
import ast
import base64
import io
import cv2
import base64 
import numpy as np
from PIL import Image
import json
import argparse
import time
parser = argparse.ArgumentParser()

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# api-endpoint
with open('config.json') as f:
    data = json.load(f)
    URL = data["url"]
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
NUM_CLASSES = 2
VALID_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png"
]

# extracting labels from the file
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def getResults(requestData):
    # Takes the image as a input and processes the output with the specifies URL
    imgdata = base64.b64decode(requestData["data"])
    data = Image.open(io.BytesIO(imgdata))
    image = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2RGB)
    [h, w] = image.shape[:2]
    
    headers = {'Accept': 'application/octet-stream',
            'content-type': 'application/json'}
    start_time1 = time.time()
    r = requests.post(url = URL , data = json.dumps(requestData), headers = headers)
    elapsed_time1 = time.time() - start_time
    print("Time to get the results : ", elapsed_time1)
    data = r.json()
    data1 = ast.literal_eval(data)

    # visulaization of the results
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(data1["boxes"]),
            np.squeeze(data1["classes"]).astype(np.int32),
            np.squeeze(data1["scores"]),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

    cv2.namedWindow("Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Results", w,h)
    cv2.imshow("Results", image)
    k = cv2.waitKey(4000) & 0xff
    
if __name__ == "__main__":
    parser.add_argument('-u','--url', type=str, help='Image url', required=True)
    args = parser.parse_args()
    for e in VALID_IMAGE_EXTENSIONS:
        if(args.url.endswith(e)):
            start_time = time.time()
            requestData = {"data": base64.b64encode(requests.get(args.url).content).decode("utf-8")}
            elapsed_time = time.time() - start_time
            print("Time to convert the image in base64 : ", elapsed_time)
            getResults(requestData)
    
    